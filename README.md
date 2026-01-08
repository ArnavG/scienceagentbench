# A4 — ScienceAgentBench (ClinTox)

**CNET ID:** arnavgurudatt

## Project Description

This project implements and evaluates an **AI-agentic scientific assistant** for a chemistry-focused data science workflow using a task from **ScienceAgentBench**. The core benchmark problem is to train a **multitask molecular classifier** on the **ClinTox** dataset to predict **clinical toxicity (CT_TOX)** and **FDA approval (FDA_APPROVED)** from a compound’s **SMILES** representation, then produce and save **test-set probability predictions** alongside evaluation metrics.

I approached the task in three stages. First, I completed the workflow manually using standard cheminformatics/ML tooling: loading the dataset, featurizing molecules with **ECFP (circular fingerprints)**, training a **DeepChem MultitaskClassifier**, and recording performance under strong class imbalance. Second, I tested a **one-shot ReAct-style agent** that uses tool-calling with a Python execution environment to automate the same pipeline end-to-end, and analyzed common failure modes (dependency friction, brittle featurization/model APIs, and evaluation reliability). Third, I built a **multi-agent system** that decomposes the workflow into specialized roles—**Featurizer → Modeler → Evaluator**—so each agent has a narrow, verifiable responsibility and passes explicit artifacts (cached features, prediction CSVs, metric CSVs) downstream. This decomposition improved robustness versus one-shot execution by reducing cascading errors and forcing evaluation to be computed from saved outputs rather than generated text.

This repo contains my work for **ScienceAgentBench**. I focus on a single benchmark instance centered on the **ClinTox** dataset: training models to predict **(1) clinical toxicity** and **(2) FDA approval** from molecular structure (SMILES), then saving test predictions and evaluation metrics.

I implement and compare:
1. **Manual completion** (DeepChem multitask model)
2. **One-shot completion** (ReAct-style agent using a Python REPL tool)
3. **Multi-agent system** (specialized agents for featurization → modeling → evaluation)
4. **Bonus:** human-in-the-loop run on a second ScienceAgentBench task (DKPES)

---

## Repository Layout

Typical structure (key files / folders):

- `a4_report.pdf` (or `a4_report.md`) — writeup of experiments and results.
- `one_shot.py` (and/or `one_shot_prompts/`, `one_shot_logs/`) — one-shot agent implementation & traces.
- `multi_agent.py` — multi-agent pipeline (featurizer agent, modeling agent, evaluator agent).
- `data/`
  - `clintox.csv` — ClinTox dataset (SMILES + labels)
  - *(optional)* `clintox_fp.csv` — cached featurized fingerprints (saved by featurizer agent as a fail-safe).
- `agent_predictions/` — outputs from the one-shot/ReAct agent run (predictions + metrics).
- `multiagent_predictions/` — outputs from the multi-agent run (predictions + metrics).
- `pred_results/` — outputs for the bonus DKPES run (predictions + metrics).

---

## Task Chosen (ClinTox)

> “Train a multitask model on the Clintox dataset to predict a drug's toxicity and FDA approval status. Save the test set predictions, including the SMILES representation of drugs and the probability of positive labels, to `pred_results/clintox_test_pred.csv`.”

### Domain constraints (from the benchmark)
- **Featurization:** Extended-Connectivity Fingerprints (ECFPs) via DeepChem’s `CircularFingerprint` featurizer.
- **Model:** DeepChem `MultitaskClassifier` with two binary heads (FDA approval + clinical toxicity).

---

## Environment / Dependencies

You’ll need a Python environment with the typical ML stack plus chemistry tooling.

Suggested:
- `python>=3.10`
- `deepchem`
- `rdkit`
- `numpy`, `pandas`
- `scikit-learn`
- *(optional, depending on your DeepChem install)* `torch`

If you use a conda environment, make sure RDKit is installed via conda-forge.

---

## How to Run

### 1) Manual ClinTox (DeepChem MultitaskClassifier)
My manual workflow:
- Load `data/clintox.csv`
- Featurize SMILES → ECFP (2048 bits)
- Train DeepChem `MultitaskClassifier` with class-weighting to address imbalance
- Evaluate & record metrics

The report documents the approach and the exact architecture/hyperparameters used.

### 2) One-shot Agent (ReAct + Python REPL Tool)
Run:
```bash
python one_shot.py
```
This implements a tool-using agent loop (**LLM ↔ Python REPL**), attempting to:

- Load data  
- Featurize and/or load prepared features  
- Train classifiers  
- Save predictions + metrics under `agent_predictions/`  

In my experiments, the “raw” one-shot setup was fragile (featurization and DeepChem model calls frequently failed), so I adapted the pipeline to use **`sklearn.RandomForestClassifier`** with **pre-split train/test CSVs** for reliability. *(a4_report)*

**Outputs (examples):**
- `agent_predictions/fda_predictions.csv`  
- `agent_predictions/tox_predictions.csv`  
- `agent_predictions/fda_metrics.csv`  
- `agent_predictions/tox_metrics.csv`  

---

## 3) Multi-agent System (Featurizer → Modeler → Evaluator)

**Run:** `python multi_agent.py`

This system decomposes the ClinTox task into three specialized agents. *(a4_report)*

### (A) Featurizer Agent
- Loads `data/clintox.csv`  
- Featurizes SMILES using DeepChem `CircularFingerprint` (ECFP)  
- Filters invalid SMILES/molecules automatically  
- Builds `dc.data.NumpyDataset(X, y)`  
- Creates an 80/20 train/test split  
- Optionally saves `data/clintox_fp.csv` as a cached artifact *(a4_report)*  

### (B) Modeling Agent
- Trains a DeepChem `MultitaskClassifier`  
- Generates test-set probabilities and hard predictions  
- Saves per-task CSV predictions under `multiagent_predictions/` *(a4_report)*  

### (C) Evaluator Agent
- Loads saved predictions  
- Computes accuracy, precision/recall/F1 (both classes), and ROC AUC  
- Saves metric CSVs under `multiagent_predictions/` *(a4_report)*  

**Outputs (examples):**
- `multiagent_predictions/fda_approved_predictions.csv`  
- `multiagent_predictions/ct_tox_predictions.csv`  
- `multiagent_predictions/fda_metrics.csv`  
- `multiagent_predictions/tox_metrics.csv`  

---

## Results Summary (High-level)

### Manual (DeepChem MultitaskClassifier)
- Strong overall accuracy (~0.94) on both tasks, but minority-class performance is the main challenge due to severe class imbalance. *(a4_report)*

### One-shot Agent (ReAct)
- Initially brittle when required to (a) featurize SMILES and (b) train DeepChem models directly.  
- Became reliable after switching to a simpler sklearn RF setup using pre-saved train/test data and a more explicit system prompt that enforced correct tool usage (e.g., always `print()` inside REPL). *(a4_report)*

### Multi-agent System
Made clear progress over one-shot by isolating responsibilities:
- Featurization became consistently successful  
- DeepChem `MultitaskClassifier` training and prediction succeeded  
- Evaluation metrics were computed from real saved outputs (reducing hallucination risk) *(a4_report)*  

---

## Bonus: Human-in-the-loop (DKPES)

I also tested an agent-assisted run on a separate benchmark:

- DKPES: RandomForest classifier predicting signal inhibition; choose a threshold to convert the target into binary labels.

**Outputs saved to:**
- `pred_results/dkpes_test_pred.csv`  
- `pred_results/dkpes_metrics.csv`  

---

## Notes / Common Issues

- Class imbalance is the central difficulty for ClinTox (both `FDA_APPROVED` and `CT_TOX` are highly skewed). *(a4_report)*  
- One-shot tool agents can fail for “small” reasons (missing `print()`, dependency mismatches, brittle DeepChem imports).  
- Multi-agent decomposition improves robustness by narrowing each agent’s scope and reducing cascading failures. *(a4_report)*  

---

## References

- ScienceAgentBench paper: https://arxiv.org/abs/2410.05080  
- DeepChem documentation (`MultitaskClassifier`, `CircularFingerprint` / ECFP)