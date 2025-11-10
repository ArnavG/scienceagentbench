import os
import pandas as pd

from inference_auth_token import get_access_token

from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


def run_agent(role_name: str, system_prompt: str, user_prompt: str, llm_with_tools, tool_map, max_steps: int = 20):
    
    print(f"\n================ {role_name.upper()} AGENT START ================\n")

    system_msg = SystemMessage(content=system_prompt)
    user_msg = HumanMessage(content=user_prompt)
    messages = [system_msg, user_msg]

    for step in range(max_steps):
        print(f"\n--- {role_name} | LLM STEP {step + 1} ---\n")
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
            continue

        # If there are no tool calls available print the result and exit the loop
        else:
            print(f"\n=== {role_name} AGENT FINAL MESSAGE ===\n")
            print(ai_msg.content)
            print(f"\n================ {role_name.upper()} AGENT END ================\n")
            return ai_msg.content

    print(f"\n{role_name}: Reached maximum number of steps without final answer.")
    if messages and hasattr(messages[-1], "content"):
        print("Last model message:\n", messages[-1].content)
    return None


def main():
    os.makedirs("multiagent_predictions", exist_ok=True)

    sab_df = pd.read_csv("data/ScienceAgentBench.csv")
    task_row = sab_df.loc[sab_df["instance_id"] == 1].iloc[0]

    task_inst = task_row["task_inst"]
    domain_knowledge = task_row["domain_knowledge"]

    access_token = get_access_token()

    python_repl = PythonREPLTool()
    tools = [python_repl]
    tool_map = {python_repl.name: python_repl}

    llm = ChatOpenAI(
        model="gpt-oss-120b-131072",
        base_url="https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
        api_key=access_token,
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(tools)

    # ---------------------------------------------------------------------
    # Featurizer Agent
    # ---------------------------------------------------------------------
    featurizer_system = """
    You are the FEATURIZER agent in a multi-agent system.
    You work in a local Python environment and have access to a Python REPL tool called `Python_REPL`
    that can execute arbitrary Python code. Your sole responsibility is to:

    1. Load the Clintox dataset:
    - The data file is "data/clintox.csv".
    - It contains three relevant columns:
        - "smiles": SMILES string for each compound
        - "FDA_APPROVED": binary label (0/1)
        - "CT_TOX": binary label (0/1)

    2. Featurize molecules:
    - Use DeepChem's circular fingerprint featurizer to convert SMILES into fixed-length fingerprint vectors.
    - For example, you can use a 2048-bit circular (ECFP-like) fingerprint with a reasonable radius.
    - You MUST ensure:
        - Only molecules that successfully featurize into FULL-length vectors (e.g., length 2048) are kept.
        - Any invalid or failed SMILES are filtered out.

    Hints (do NOT just paste this as-is; implement your own code):
    - Import DeepChem and RDKit inside Python_REPL.
    - Create a CircularFingerprint featurizer from DeepChem.
    - Call `.featurize` on the SMILES column to get a list/array of fingerprints.
    - Filter to only those whose shape matches the expected fingerprint size.
    - Stack them into a 2D NumPy array of shape (num_valid_samples, num_features).
    - Build a pandas DataFrame with feature columns named like "fp_0", "fp_1", ..., and attach labels.

    3. Build datasets:
    - Construct a combined pandas DataFrame `clintox_fp_df` containing:
        - All float-like feature columns,
        - The two label columns "FDA_APPROVED" and "CT_TOX".
    - Create NumPy arrays:
        - `X` = feature matrix (float32)
        - `y` = label matrix with shape (N, 2) for the two tasks (float32)
    - Then build a DeepChem dataset:
        - `dataset = dc.data.NumpyDataset(X, y)`

    4. Train/test split:
    - Use DeepChem's `RandomSplitter` (or equivalent) to create:
        - `train_dataset`
        - `test_dataset`
    - You should also keep references to `X`, `y`, and any useful train/test splits (e.g. by indices)
        in global variables so that later agents (Modeling and Evaluator) can reuse them.

    5. Persist some artifacts:
    - Optionally save a featurized CSV version (with features + labels) to disk, such as:
        - "data/clintox_fp.csv"
    - This is optional, but can help later agents if needed.

    Execution rules:
    - ALWAYS run Python code via the `Python_REPL` tool.
    - Before doing RDKit work, disable RDKit logs to keep the console clean:

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    print("RDKit logger disabled")

    - Print brief status messages (e.g., shapes of arrays, number of valid molecules) so humans can
    inspect your progress.

    When you are done, summarize:
    - How many valid molecules you kept,
    - The shape of the feature matrix,
    - The fingerprint length,
    - That you created `dataset`, `train_dataset`, and `test_dataset` in the Python environment.
        """.strip()

    featurizer_user = f"""
    You are working on ScienceAgentBench instance 1.

    Task instruction:
    {task_inst}

    Domain knowledge (for context only, you do NOT need to restate it):
    {domain_knowledge}

    Your role as FEATURIZER:
    - Read "data/clintox.csv".
    - Featurize SMILES into fixed-length fingerprint vectors.
    - Build a DeepChem NumpyDataset with X (features) and y (two labels).
    - Split into `train_dataset` and `test_dataset`.
    - Leave these Python objects in memory for the next agents to use.

    Use `Python_REPL` for all code execution.
        """.strip()

    run_agent(
        "Featurizer",
        featurizer_system,
        featurizer_user,
        llm_with_tools,
        tool_map,
    )

    # ---------------------------------------------------------------------
    # Modeling Agent
    # ---------------------------------------------------------------------
    modeling_system = """
    You are the MODELING agent in a multi-agent system.
    A previous FEATURIZER agent has already run code in this Python environment and should have
    created the following objects (or similar):

    - `clintox_fp_df`: a pandas DataFrame with fingerprint feature columns and labels
    - `X`, `y`: NumPy arrays with features and labels
    - `dataset`: a DeepChem NumpyDataset
    - `train_dataset`, `test_dataset`: DeepChem datasets for training and testing

    Your responsibilities:

    1. Inspect the environment:
    - Use Python_REPL to inspect global variables (e.g., `globals().keys()` and print types/shapes).
    - Confirm that you have access to at least `train_dataset` and `test_dataset`, or to `X_train`, `X_test`, `y_train`, `y_test` if they exist.

    2. Choose a model:
    - You may choose ANY reasonable classifier:
        - scikit-learn models such as `LogisticRegression` or `RandomForestClassifier`, or
        - DeepChem models such as `MultitaskClassifier`.
    - If you choose DeepChem, here is example usage for reference (adapt it to the actual variables):

    import torch
    import deepchem as dc
    from deepchem.models.fcnet import MultitaskClassifier

    model = dc.models.MultitaskClassifier(
        n_tasks, # number of prediction tasks you are doing
        n_features, # number of features in the training and test data
        n_classes, # number of classes per prediction class
        layer_sizes=[...], # a list of integers that correspond to the size of each hidden layer in the classifier
        dropouts=0.1, # a float value indicating how many nodes to dropout during training; default value 0.1, can be changed
        learning_rate=0.001, # default value, can change
        batch_size=64, # default value, can change
        verbosity="high",
    )
    model.fit(train_dataset, nb_epoch) # can change the number of epochs (nb_epoch) you train over
    train_pred = model.predict(train_dataset)
    test_pred = model.predict(test_dataset)

    - You may adapt this to your actual setup from the FEATURIZER.

    3. Train the model:
    - Fit your chosen model on the training data.
    - Then generate predictions on the test data for BOTH tasks:
        - For DeepChem MultitaskClassifier: use `model.predict(test_dataset)` to get probabilities
        of shape (N_test, n_tasks, n_classes).
        - Convert to:
        - class probabilities for the positive class (label 1),
        - hard class predictions (0/1) using a threshold such as 0.5.

    4. Save predictions for the Evaluator agent:
    - You MUST write prediction CSVs to the "multiagent_predictions/" folder.
    - Create two CSV files:
        - "multiagent_predictions/fda_predictions.csv"
        - "multiagent_predictions/tox_predictions.csv"
    - Each CSV should contain at least:
        - "id": an integer index or row ID,
        - "y_true": the true label for that task on the test set,
        - "y_pred": the predicted label (0/1),
        - "prob_pos": the predicted probability of the positive class (1).
    - You may also include additional columns if you like.

    Execution rules:
    - ALWAYS use `Python_REPL` to execute code.
    - Print brief logs (shapes of arrays, model training status, etc.) so progress is visible.

    When you are done, summarize:
    - What model you chose,
    - How many test samples you predicted on,
    - The locations of the saved prediction CSVs.
        """.strip()

    modeling_user = """
    Your role as MODELING agent:

    - Reuse the featurized data and datasets created by the FEATURIZER agent in this Python environment.
    - Train a suitable model (e.g., DeepChem MultitaskClassifier or an sklearn classifier).
    - Predict on the test set for both tasks (FDA_APPROVED and CT_TOX).
    - Save prediction CSVs as:
    - multiagent_predictions/fda_predictions.csv
    - multiagent_predictions/tox_predictions.csv
    with columns at least ["id", "y_true", "y_pred", "prob_pos"].

    Use the Python_REPL tool for all code execution.
        """.strip()

    run_agent(
        "Modeling",
        modeling_system,
        modeling_user,
        llm_with_tools,
        tool_map,
    )

    # ---------------------------------------------------------------------
    # Evaluator Agent
    # ---------------------------------------------------------------------
    evaluator_system = """
    You are the EVALUATOR agent in a multi-agent system.
    Previous agents have:

    - Featurized the Clintox dataset and created train/test splits.
    - Trained a model and written prediction CSVs to "multiagent_predictions/":
    - "multiagent_predictions/fda_predictions.csv"
    - "multiagent_predictions/tox_predictions.csv"

    Each prediction CSV is expected to contain at least:
    - "id": integer index,
    - "y_true": true label (0 or 1),
    - "y_pred": predicted class (0 or 1),
    - "prob_pos": predicted probability of class 1.

    Your responsibilities:

    1. Load the prediction CSVs:
    - Read "multiagent_predictions/fda_predictions.csv" and "multiagent_predictions/tox_predictions.csv"
        using pandas.

    2. Compute evaluation metrics:
    - For each task (FDA_APPROVED and CT_TOX), compute:
        - accuracy
        - precision for both classes 0 and 1
        - recall for both classes 0 and 1
        - F1 score for both classes 0 and 1
        - ROC AUC (binary)
    - You may use sklearn.metrics functions:
        - accuracy_score
        - precision_score
        - recall_score
        - f1_score
        - roc_auc_score

    - Construct a long-form pandas DataFrame `metrics_df` with columns:
        - "task": e.g., "FDA_APPROVED" or "CT_TOX"
        - "metric_name": "accuracy", "precision", "recall", "f1", "roc_auc"
        - "class_label": 0, 1, or "overall" for metrics that do not depend on a specific class
        - "value": numeric value of the metric

    3. Save metrics:
    - Save two CSV files under "multiagent_predictions/":
        - "multiagent_predictions/fda_metrics.csv" containing only rows where task == "FDA_APPROVED"
        - "multiagent_predictions/tox_metrics.csv" containing only rows where task == "CT_TOX"

    4. (Optional) Print a brief summary of the main metrics (e.g., accuracy and ROC AUC for each task)
    so humans can quickly inspect performance.

    Execution rules:
    - ALWAYS use `Python_REPL` to execute code.
    - Print brief logs when loading files and computing metrics.

    When you are done, summarize:
    - Where you saved the metrics CSVs,
    - The main metric values for each task.
        """.strip()

    evaluator_user = """
    Your role as EVALUATOR agent:

    - Load the prediction CSVs created by the MODELING agent:
    - multiagent_predictions/fda_predictions.csv
    - multiagent_predictions/tox_predictions.csv
    - Compute accuracy, precision, recall, F1 (for both classes), and ROC AUC for each task.
    - Save metrics as:
    - multiagent_predictions/fda_metrics.csv
    - multiagent_predictions/tox_metrics.csv

    Use Python_REPL for all code execution and print a brief summary of results.
        """.strip()

    run_agent(
        "Evaluator",
        evaluator_system,
        evaluator_user,
        llm_with_tools,
        tool_map,
    )

    print("\nMulti-agent pipeline completed.\n")
    print("Check the 'multiagent_predictions/' folder for prediction and metric CSVs.")


if __name__ == "__main__":
    main()