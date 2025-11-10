import os
import pandas as pd

from inference_auth_token import get_access_token

from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


def main():
    os.makedirs("agent_predictions", exist_ok=True)

    df = pd.read_csv("data/ScienceAgentBench.csv")
    task_row = df.loc[df["instance_id"] == 1].iloc[0]

    task_inst = task_row["task_inst"]
    domain_knowledge = "None. " #task_row["domain_knowledge"]

    access_token = get_access_token()

    # Python REPL tool that runs arbitrary Python code in this repo
    python_repl = PythonREPLTool()
    tools = [python_repl]
    tool_map = {python_repl.name: python_repl}

    system_prompt = """
    You are an autonomous machine learning agent working in a local Python environment.
    You have access to a Python REPL tool called `Python_REPL` that can execute arbitrary
    Python code in this repository. You will use this tool to load data, train models,
    compute metrics, and save CSV files.

    IMPORTANT:
    - You MUST use the `Python_REPL` tool to actually run code.
    - Do NOT just print code as text; always execute via the tool.
    - When calling the tool, always provide a single argument "query" containing the Python code.

    Your goals:

    1. Data:
    - The training and test data are stored in the following CSV files (relative to the current directory):
        - "data/train_fda.csv"  (has column "FDA_APPROVED" as the target)
        - "data/test_fda.csv"   (has column "FDA_APPROVED" as the target)
        - "data/train_tox.csv"  (has column "CT_TOX" as the target)
        - "data/test_tox.csv"   (has column "CT_TOX" as the target)

    2. For each task (FDA_APPROVED and CT_TOX), you MUST:
    a. Load the corresponding train and test data from the CSV files.
    b. Fit an appropriate classification model on the training data.
    - You may use scikit-learn models such as RandomForestClassifier,
        LogisticRegression, etc.
    c. Use the trained model to generate predictions on the test data.
    d. Save the test predictions as a CSV file under the "agent_predictions/" folder.
    - Include at least: an ID or index and the predicted class
        (and optionally probability).
    - Use filenames such as:
        - "agent_predictions/fda_predictions.csv"
        - "agent_predictions/tox_predictions.csv"
    e. Compute and report the following evaluation metrics on the test set:
    - accuracy
    - precision for both the positive and negative classes
    - recall for both the positive and negative classes
    - F1 score for both the positive and negative classes
    - ROC AUC
    f. Save these evaluation metrics as a CSV file under the "agent_predictions/" folder.
    - Use filenames such as:
        - "agent_predictions/fda_metrics.csv"
        - "agent_predictions/tox_metrics.csv"
    - Include columns like:
        - "task" (e.g. "FDA_APPROVED" or "CT_TOX")
        - "metric_name"
        - "class_label" (for metrics that depend on a class,
            e.g. precision/recall/F1)
        - "value"

    3. Execution:
    - Any time you need to load data, train, evaluate, or save CSVs,
    you MUST call the `Python_REPL` tool with valid Python code.
    - If you encounter errors, inspect the traceback and fix your code
    with another `Python_REPL` tool call.

    4. Final output:
    - After you have successfully trained models, saved prediction CSVs,
    and saved metric CSVs, write a concise natural language summary describing:
    - What models you used,
    - Where the prediction and metric CSV files were saved,
    - The main performance metrics (e.g. test accuracy and ROC AUC for each task).

    Think step-by-step, using multiple `Python_REPL` tool calls as needed
    to complete the full pipeline.
    Do NOT stop after only generating code as a string; you must actually run the code,
    train the models, and write the CSV files under agent_predictions/.
    """.strip()

    llm = ChatOpenAI(
        model="gpt-oss-120b-131072",
        base_url="https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
        api_key=access_token,
        temperature=0.1,
    )

    llm_with_tools = llm.bind_tools(tools)
    system_msg = SystemMessage(content=system_prompt)

    user_prompt = f"""
    You are working on ScienceAgentBench instance 1.

    Task instruction:
    {task_inst}

    Domain knowledge that may be helpful:
    {domain_knowledge}

    Please complete the full pipeline described in the system message for BOTH tasks:
    - FDA_APPROVED (using train_fda.csv / test_fda.csv)
    - CT_TOX (using train_tox.csv / test_tox.csv)

    Use the Python_REPL tool to:
    - Load the data,
    - Train classification models,
    - Evaluate them on the test sets,
    - Save predictions as CSV files in the "agent_predictions/" folder,
    - Save evaluation metrics (accuracy, precision, recall, F1 for both classes, and ROC AUC)
    as CSV files in the "agent_predictions/" folder,
    - And then summarize what you did and the main metrics.
    """.strip()

    messages = [
        system_msg,
        HumanMessage(content=user_prompt),
    ]
    max_steps = 20
    for step in range(max_steps):
        print(f"\n--- LLM STEP {step + 1} ---\n")
        ai_msg = llm_with_tools.invoke(messages)

        # Sanitize tool_calls so that args are JSON-serializable
        if getattr(ai_msg, "tool_calls", None):
            sanitized_tool_calls = []
            for tc in ai_msg.tool_calls:
                raw_args = tc.get("args", {}) or {}
                sanitized_args = {}
                if isinstance(raw_args, dict) and "query" in raw_args:
                    sanitized_args["query"] = raw_args["query"]
                sanitized_tool_calls.append(
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "type": tc.get("type", "function"),
                        "args": sanitized_args,
                    }
                )
            ai_msg.tool_calls = sanitized_tool_calls

        messages.append(ai_msg)

        # If the model calls tools, execute them
        if getattr(ai_msg, "tool_calls", None):
            for tc in ai_msg.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {}) or {}
                tool_id = tc["id"]

                if tool_name not in tool_map:
                    tool_output = (
                        f"Error: unknown tool '{tool_name}'. "
                        f"Available tools: {list(tool_map.keys())}"
                    )
                else:
                    query = tool_args.get("query")
                    if not isinstance(query, str):
                        tool_output = (
                            "Error: missing or invalid 'query' argument for Python_REPL. "
                            "Please provide code as the 'query' string."
                        )
                    else:
                        tool = tool_map[tool_name]
                        try:
                            tool_output = tool.invoke({"query": query})
                        except Exception as e:
                            tool_output = f"Error while running tool {tool_name}: {e}"

                print(
                    f"[TOOL {tool_name}] args={tool_args}\nOUTPUT:\n{tool_output}\n"
                )

                # Feed tool result back to the model
                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_id,
                    )
                )
            continue # Continue loop: model sees tool outputs and can call more tools or answer

        # If there are no tool calls available print the result and exit the loop
        else:
            print("\n=== FINAL AGENT RESULT ===\n")
            print(ai_msg.content)
            return

    # If we exit the loop without a final answer
    print("\nReached maximum number of steps without final answer.")
    if messages and hasattr(messages[-1], "content"):
        print("Last model message:\n", messages[-1].content)


if __name__ == "__main__":
    main()