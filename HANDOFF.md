# WMJailbreak Server Handoff

This document summarizes the current state of the WMJailbreak project and the next steps for continuing experiments on a server.

## 1. Research Goal

The project is about **attack-side multi-turn jailbreak planning with a world model**.

Working idea:

- Treat a multi-turn jailbreak conversation as a sequence of state transitions.
- Build a safety representation space from the conversation history, attacker action, target response, refusal/compliance signals, and harm/risk signals.
- Train a Dreamer/DreamerV3-style latent world model to predict how the target model's safety state evolves across turns.
- Use the learned world model for active attack planning: given the first few failed or partial jailbreak attempts, imagine future trajectories and choose the next red-team action that maximizes eventual unsafe compliance.

The emphasis is **attack**, not defense. The novelty should be framed as:

- A world model used as an **active jailbreak planner**, not merely a safety detector.
- Multi-turn attack trajectories represented as Markov-style transitions.
- Planning in a learned safety/risk latent space instead of directly optimizing a single jailbreak prompt.
- Compatibility with classifier and LLM judge signals for reward construction.

## 2. Current Repository State

GitHub repository:

- `https://github.com/axbhb/WMJailbreak.git`

Current pushed branch:

- `main`

Important tracked files:

- `README.md`
- `requirements.txt`
- `data/README.md`
- `scripts/convert_xguard_to_transitions.py`
- `scripts/download_evaluator_models.py`
- `scripts/annotate_transitions_with_classifiers.py`
- `scripts/repair_jsonl_objects.py`
- `HANDOFF.md`

Large data/model artifacts are intentionally **not committed**.

## 3. Completed Work

### 3.1 XGuard-Train Downloaded Locally

Local raw dataset path on the Windows machine:

- `D:\WMjailbreak\data\xguard-train\xguard-train.json`

Dataset structure:

- Top-level object: list
- Each item has a `conversations` field
- Conversation messages have:
  - `from`: `human` or `gpt`
  - `value`: message text

Observed dataset size:

- Conversations: `30,695`
- User-assistant transition pairs: `156,385`

Conversation turn distribution:

| Turns | Conversations |
| ---: | ---: |
| 1 | 569 |
| 2 | 3,585 |
| 3 | 4,617 |
| 4 | 3,891 |
| 5 | 2,627 |
| 6 | 1,746 |
| 7 | 13,660 |

### 3.2 XGuard Converted to Dreamer-Style Transition Skeleton

Generated file:

- `D:\WMjailbreak\data\xguard-train\transitions\xguard_transitions.jsonl`

Line count:

- `156,385`

The conversion script creates one transition per user-assistant pair. Each transition includes:

- `dialogue_id`
- `turn`
- `history`
- `user_prompt`
- `assistant_response`
- `response_type`
- `refusal_score`
- `abstract_action`
- placeholder fields for later labels:
  - `risk_score`
  - `partial_compliance_score`
  - `unsafe_compliance_score`
  - `future_boundary`
  - `turns_to_boundary`
  - `reward_value`

Conversion command:

```bash
python scripts/convert_xguard_to_transitions.py \
  --input data/xguard-train/xguard-train.json \
  --output data/xguard-train/transitions/xguard_transitions.jsonl
```

### 3.3 Evaluator Weights Downloaded Locally

Requirement from the local run:

- Model weights should be stored under `F:\WMjailbreak`, not under the default C drive cache.

Local Hugging Face cache path:

- `F:\WMjailbreak\hf_cache`

Manifest:

- `F:\WMjailbreak\evaluator_model_paths.json`

Downloaded models:

- WildGuard: `allenai/wildguard`
- HarmBench classifier: `cais/HarmBench-Llama-2-13b-cls`

Local snapshot paths:

- WildGuard:
  - `F:\WMjailbreak\hf_cache\models--allenai--wildguard\snapshots\cbba4823f3e8020e5a74a5e29bf85072def6f2ff`
- HarmBench:
  - `F:\WMjailbreak\hf_cache\models--cais--HarmBench-Llama-2-13b-cls\snapshots\bda705349d1144fa618770bea64d99ce54e3835b`

Download command:

```bash
python scripts/download_evaluator_models.py \
  --cache-dir /path/to/hf_cache \
  --models wildguard harmbench
```

On the original Windows machine, `/path/to/hf_cache` was:

```text
F:\WMjailbreak\hf_cache
```

### 3.4 WildGuard Full Annotation Completed Locally

WildGuard has been run on all `156,385` transitions.

Local final clean file:

- `D:\WMjailbreak\data\xguard-train\transitions\xguard_transitions_wildguard.clean.jsonl`

Validation:

- Lines: `156,385`
- Unique `(dialogue_id, turn)` pairs: `156,385`
- WildGuard records: `156,385`
- File size: about `2.57 GB`

Important note:

- The first raw WildGuard output had one malformed concatenated JSONL line and one duplicate due to interruption.
- It was repaired and deduplicated with `scripts/repair_jsonl_objects.py`.
- The clean final artifact is `xguard_transitions_wildguard.clean.jsonl`.

WildGuard command template:

```bash
python scripts/annotate_transitions_with_classifiers.py \
  --input data/xguard-train/transitions/xguard_transitions.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard.jsonl \
  --evaluator wildguard \
  --model-cache-dir /path/to/hf_cache \
  --batch-size 8 \
  --flush-every 1000 \
  --max-input-tokens 2048 \
  --max-new-tokens 16 \
  --resume
```

Repair command:

```bash
python scripts/repair_jsonl_objects.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl
```

### 3.5 HarmBench Smoke Test Completed, Full Run Not Completed

HarmBench weights were downloaded successfully.

Smoke/benchmark status:

- 1-item smoke test succeeded.
- 20-item benchmark succeeded.
- Full HarmBench annotation over all `156,385` transitions has **not** been completed.

Observed local performance:

- Machine: RTX 4090 24 GB
- HarmBench 13B needed CPU offload on this machine.
- Model loading took around 7.5-8 minutes.
- Inference was around 3 seconds per item in a 20-item benchmark.
- Full 156K annotation would take days on the local 4090, so this should be moved to a stronger server GPU if available.

Important caveat:

- XGuard does not provide an explicit HarmBench `behavior` field.
- The current HarmBench evaluator uses the current user turn as `behavior` and the previous dialogue history as `context`.
- This is acceptable as a first pass, but should be described as an approximation in experiments.

Recommended HarmBench command on server:

```bash
python scripts/annotate_transitions_with_classifiers.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.jsonl \
  --evaluator harmbench \
  --model-cache-dir /path/to/hf_cache \
  --batch-size 1 \
  --flush-every 100 \
  --max-input-tokens 1536 \
  --max-history-chars 2000 \
  --max-user-chars 1000 \
  --max-response-chars 2000 \
  --max-new-tokens 4 \
  --resume
```

After the run:

```bash
python scripts/repair_jsonl_objects.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.clean.jsonl
```

If the server has a large-memory GPU, try increasing `--batch-size` to `2` or `4`.

## 4. Files Not Included in Git

The following are intentionally ignored and must be recreated, downloaded, or copied manually:

- Raw XGuard dataset:
  - `data/xguard-train/xguard-train.json`
- Converted transitions:
  - `data/xguard-train/transitions/xguard_transitions.jsonl`
- WildGuard outputs:
  - `data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl`
- HarmBench outputs and benchmarks
- Logs
- Reference PDFs
- Local model weights/cache:
  - `F:\WMjailbreak\hf_cache`

Approximate local sizes:

- `xguard-train.json`: about `674 MB`
- `xguard_transitions.jsonl`: about `2.54 GB`
- `xguard_transitions_wildguard.clean.jsonl`: about `2.57 GB`
- Evaluator model cache: about `40 GB`

## 5. Server Setup Checklist

### 5.1 Clone the Repository

```bash
git clone https://github.com/axbhb/WMJailbreak.git
cd WMJailbreak
```

### 5.2 Prepare Python Environment

Install dependencies from `requirements.txt`, but install PyTorch according to the server CUDA version.

Typical packages needed:

- `torch`
- `transformers`
- `accelerate`
- `huggingface_hub`
- `datasets`
- `sentencepiece`
- `tqdm`

Example:

```bash
pip install -r requirements.txt
```

If PyTorch/CUDA is mismatched, reinstall PyTorch using the official command for the server CUDA version.

### 5.3 Hugging Face Access

Required access:

- `marslabucla/XGuard-Train`
- `allenai/wildguard`
- `cais/HarmBench-Llama-2-13b-cls`

Set token on server:

```bash
export HF_TOKEN="hf_your_token_here"
```

Then login if preferred:

```bash
huggingface-cli login
```

### 5.4 Download XGuard-Train

Example server command:

```bash
python - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

target = Path("data/xguard-train/xguard-train.json")
target.parent.mkdir(parents=True, exist_ok=True)
src = hf_hub_download(
    repo_id="marslabucla/XGuard-Train",
    filename="xguard-train.json",
    repo_type="dataset",
)
shutil.copyfile(src, target)
print(target)
PY
```

If the local clean WildGuard file can be copied to the server, copy this file directly to save time:

```text
data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl
```

Otherwise, rerun conversion and WildGuard annotation on server.

### 5.5 Convert XGuard to Transitions

```bash
python scripts/convert_xguard_to_transitions.py \
  --input data/xguard-train/xguard-train.json \
  --output data/xguard-train/transitions/xguard_transitions.jsonl
```

Expected output:

- `156,385` JSONL lines

### 5.6 Download Evaluator Models

Choose a large disk path for model cache, for example:

```bash
mkdir -p /data/WMjailbreak/hf_cache
python scripts/download_evaluator_models.py \
  --cache-dir /data/WMjailbreak/hf_cache \
  --models wildguard harmbench
```

Use the same cache path in later annotation commands:

```bash
--model-cache-dir /data/WMjailbreak/hf_cache
```

### 5.7 Run or Reuse WildGuard

If `xguard_transitions_wildguard.clean.jsonl` was copied from local, skip this step.

Otherwise:

```bash
python scripts/annotate_transitions_with_classifiers.py \
  --input data/xguard-train/transitions/xguard_transitions.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard.jsonl \
  --evaluator wildguard \
  --model-cache-dir /data/WMjailbreak/hf_cache \
  --batch-size 8 \
  --flush-every 1000 \
  --max-input-tokens 2048 \
  --max-new-tokens 16 \
  --resume

python scripts/repair_jsonl_objects.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl
```

### 5.8 Run HarmBench Full Annotation

This is the immediate next major experiment.

Recommended conservative command:

```bash
python scripts/annotate_transitions_with_classifiers.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.jsonl \
  --evaluator harmbench \
  --model-cache-dir /data/WMjailbreak/hf_cache \
  --batch-size 1 \
  --flush-every 100 \
  --max-input-tokens 1536 \
  --max-history-chars 2000 \
  --max-user-chars 1000 \
  --max-response-chars 2000 \
  --max-new-tokens 4 \
  --resume
```

If the server GPU has enough VRAM, increase:

- `--batch-size 2`
- `--batch-size 4`

After completion:

```bash
python scripts/repair_jsonl_objects.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.clean.jsonl
```

## 6. Next Research/Engineering Steps

### Step 1: Finish HarmBench Full Annotation

Target output:

```text
data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.clean.jsonl
```

Validation criteria:

- `156,385` lines
- Unique `(dialogue_id, turn)` pairs: `156,385`
- Every record has WildGuard fields
- Every record has HarmBench fields

### Step 2: Implement Label Fusion and Reward Construction

Implemented script:

```text
scripts/build_dreamer_rewards.py
```

Inputs:

```text
data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.clean.jsonl
```

Outputs:

```text
data/xguard-train/transitions/xguard_transitions_rewards.jsonl
```

Suggested fields to fill:

- `risk_score`
- `unsafe_compliance_score`
- `partial_compliance_score`
- `refusal_score`
- `risk_delta`
- `future_boundary`
- `turns_to_boundary`
- `reward_value`

Initial reward idea:

```text
reward_value =
  + alpha * unsafe_compliance_score
  + beta  * partial_compliance_score
  + gamma * risk_delta
  - lambda * refusal_score
```

Trajectory-level future labels:

- `future_boundary = 1` if a later turn in the same dialogue reaches unsafe compliance or high risk.
- `turns_to_boundary = number of turns until the first future high-risk/unsafe-compliance state`.

Need to tune thresholds after inspecting score distributions.

### Step 3: Split Train/Validation/Test by Dialogue ID

Avoid leakage across turns from the same conversation.

Implemented script:

```text
scripts/split_transitions.py
```

Recommended split:

- Train: 80%
- Validation: 10%
- Test: 10%

Split unit:

- `dialogue_id`, not individual transition lines.

### Step 4: Train the World Model

Initial model target:

- Encode transition state from:
  - dialogue history
  - current user prompt
  - assistant response
  - classifier/judge scores
- Predict:
  - next latent state
  - next risk/refusal/unsafe compliance
  - future boundary probability
  - reward

First practical implementation can be simpler than full DreamerV3:

- Text encoder: frozen sentence embedding model or small transformer encoder.
- Latent dynamics: GRU/Transformer over turns.
- Heads:
  - risk score regression
  - refusal score regression
  - unsafe compliance classification/regression
  - future boundary classification
  - reward prediction

After this works, add Dreamer-style imagination/planning.

### Step 5: Add Attack Planner

Planner input:

- Current dialogue prefix
- Current safety state estimate
- Candidate abstract red-team actions

Planner output:

- Next attacker action type or prompt rewrite strategy

Candidate action abstraction already exists heuristically in the transition conversion script. It can be expanded into categories such as:

- direct harmful request
- roleplay framing
- hypothetical/fictional framing
- educational framing
- authority/expert framing
- translation/format transformation
- decomposition/subtasking
- obfuscation
- persistence/escalation
- refusal recovery

### Step 6: Baselines

Recommended initial baselines:

- Random abstract action planner
- Rule-based multi-turn planner
- LLM-only planner without world-model imagination
- Classifier-greedy planner using current risk/refusal only
- Ablation: no future-boundary label
- Ablation: WildGuard only
- Ablation: HarmBench only

Later paper-level baselines:

- IHO-style single-turn prompt optimization
- Crescendo-style multi-turn jailbreak
- Tree/beam-search red-teaming
- PAIR/TAP-style attacker LLM

## 7. Immediate Prompt for Server Codex

Use this prompt after cloning the repo on the server:

```text
Continue the WMJailbreak experiment from HANDOFF.md.

First verify whether data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl exists.
If it exists, validate that it has 156385 unique (dialogue_id, turn) records.
If it does not exist, recreate it by downloading XGuard-Train, converting it to transitions, running WildGuard, and repairing JSONL.

Then run full HarmBench annotation on xguard_transitions_wildguard.clean.jsonl using --resume.
Use the largest safe batch size for the server GPU.
After completion, repair/deduplicate the output JSONL and report line counts, unique transition counts, elapsed time, and output path.

Do not commit large datasets, annotation JSONL files, logs, or model weights.
```

## 8. Known Issues and Cautions

- Do not upload large data/model files to GitHub.
- Keep all annotation runs resumable with `--resume`.
- If a run is interrupted, use `scripts/repair_jsonl_objects.py` before continuing downstream.
- HarmBench is the current bottleneck.
- The current HarmBench behavior/context mapping is approximate because XGuard does not expose explicit harmful behavior labels.
- Full model training and Dreamer planner code has not been implemented yet.
- The current project is at the data/evaluator-preparation stage, not yet at model-training stage.
