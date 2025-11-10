import os
import pandas as pd

from inference_auth_token import get_access_token

from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


def main():
    os.makedirs("agent_predictions", exist_ok=True)

    df = pd.read_csv("data/ScienceAgentBench.csv")
    task_row = df.loc[df["instance_id"] == 5].iloc[0] # DKPES Prediction Task

    task_inst = task_row["task_inst"]

    access_token = get_access_token()

    # Python REPL tool that runs arbitrary Python code in this repo
    python_repl = PythonREPLTool()
    tools = [python_repl]
    tool_map = {python_repl.name: python_repl}

    system_prompt = """
    You are an autonomous machine learning agent working in a local Python environment.
    You have access to a Python REPL tool called `Python_REPL` that can execute arbitrary
    Python code in this repository. You will use this tool to load data, preprocess it,
    train a model, compute metrics, and save CSV files.

    IMPORTANT:
    - You MUST use the `Python_REPL` tool to actually run code.
    - Do NOT just print code as text; always execute via the tool.
    - When calling the tool, always provide a single argument "query" containing the Python code.

    Your task:

    1. Data:
    - The dataset is stored in:
        - "dkpes/dkpes.csv"
    - The target column is:
        - "Signal-inhibition"

    2. Preprocessing and features:
    - Load "dkpes/dkpes.csv".
    - Inspect the columns and decide which ones are useful features.
    - Drop clearly unnecessary or identifier-like columns (e.g., IDs or constant columns).
    - For any categorical feature columns, convert them into a numerical representation
    (for example: one-hot encoding, label encoding, or another reasonable encoding strategy).
    - Make sure the final feature matrix is purely numeric.

    3. Target and labeling:
    - Use the "Signal-inhibition" column as the target.
    - Choose a reasonable threshold on the signal inhibition values to create a binary label
    (e.g., "high inhibition" vs. "low inhibition").
    - Explain in comments (in your Python code) how you picked the threshold.
    - Train a RandomForestClassifier from scikit-learn on this binary classification task.

    4. Train/test split:
    - Create a train/test split from the dataset (for example, 80% train / 20% test).
    - Use the training split to fit the RandomForest classifier.
    - Use the test split to evaluate performance.

    5. Predictions and saving results:
    - Generate predictions on the test split.
    - Save a CSV file at:
        - "pred_results/dkpes_test_pred.csv"
    - This CSV must include at least:
        - An index or ID for each test example (e.g., the original index),
        - The predicted binary label for signal inhibition,
        - Optionally, the predicted probability for the positive class.

    6. Evaluation metrics:
    - On the test set, compute and report:
        - Accuracy
        - Precision
        - Recall
        - F1 score
        - ROC AUC
    - Print these metrics in a clear, readable format.
    - You may also save them to a separate CSV (e.g., "pred_results/dkpes_metrics.csv"),
    but printing them is required.

    7. Execution rules:
    - Any time you need to load data, preprocess, split, train, evaluate, or save CSVs,
    you MUST call the `Python_REPL` tool with valid Python code.
    - If you encounter errors, inspect the traceback and fix your code with another
    `Python_REPL` call until the full pipeline completes.

    8. Final answer:
    - After you have:
        - Loaded and preprocessed the data,
        - Created the train/test split,
        - Trained the RandomForest classifier,
        - Saved "pred_results/dkpes_test_pred.csv",
        - Computed and printed all required metrics,
    provide a short natural language summary of what you did and the main results.

    Think step-by-step, using multiple `Python_REPL` tool calls as needed to complete
    the pipeline end-to-end.
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
    You are working on ScienceAgentBench instance 5.

    Task instruction:
    {task_inst}

    Please complete the task described in the system message using the DKPES dataset.
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