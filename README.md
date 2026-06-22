# WMJailbreak

World-model-based active red-teaming pipeline for multi-turn jailbreak research.

This repository contains the code and reproducibility instructions. It intentionally does **not** commit XGuard-Train, generated JSONL annotations, logs, or evaluator weights.

## Local Artifacts

Current local artifacts were generated under:

```text
D:\WMjailbreak\data\xguard-train\
F:\WMjailbreak\hf_cache\
```

The most important completed file is:

```text
data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl
```

It contains 156,385 transitions with WildGuard annotations. HarmBench full annotation has not been completed yet.

## Setup

Create or activate a Python environment with GPU-enabled PyTorch, then install:

```bash
pip install -r requirements.txt
```

Set a Hugging Face token with access to gated assets:

```bash
export HF_TOKEN=hf_xxx
```

On Windows PowerShell:

```powershell
setx HF_TOKEN "hf_xxx"
```

You must accept access terms for:

- `marslabucla/XGuard-Train`
- `allenai/wildguard`

## Download Evaluator Models

Choose a model cache directory with enough disk space. On the original machine this was `F:/WMjailbreak/hf_cache`.

```bash
python scripts/download_evaluator_models.py \
  --cache-dir /path/to/WMJailbreak/hf_cache \
  --models wildguard harmbench
```

## Download XGuard-Train

Download the gated dataset file from Hugging Face:

```text
https://huggingface.co/datasets/marslabucla/XGuard-Train
```

Place it at:

```text
data/xguard-train/xguard-train.json
```

The file is about 0.67 GB.

## Convert XGuard to Transitions

```bash
python scripts/convert_xguard_to_transitions.py \
  --input data/xguard-train/xguard-train.json \
  --output data/xguard-train/transitions/xguard_transitions.jsonl
```

Expected output:

```text
dialogues: 30695
transitions: 156385
```

## Annotate With WildGuard

```bash
python scripts/annotate_transitions_with_classifiers.py \
  --input data/xguard-train/transitions/xguard_transitions.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard.jsonl \
  --evaluator wildguard \
  --model-cache-dir /path/to/WMJailbreak/hf_cache \
  --batch-size 8 \
  --flush-every 1000 \
  --max-input-tokens 2048 \
  --max-new-tokens 16 \
  --resume
```

If the run is interrupted, rerun the same command with `--resume`.

After completion, repair and deduplicate if needed:

```bash
python scripts/repair_jsonl_objects.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl
```

## Annotate With HarmBench

HarmBench 13B may require CPU offload on a 24GB GPU, so full annotation can be slow.

```bash
python scripts/annotate_transitions_with_classifiers.py \
  --input data/xguard-train/transitions/xguard_transitions_wildguard.clean.jsonl \
  --output data/xguard-train/transitions/xguard_transitions_wildguard_harmbench.jsonl \
  --evaluator harmbench \
  --model-cache-dir /path/to/WMJailbreak/hf_cache \
  --batch-size 1 \
  --flush-every 100 \
  --max-input-tokens 1536 \
  --max-history-chars 2000 \
  --max-user-chars 1000 \
  --max-response-chars 2000 \
  --max-new-tokens 4 \
  --resume
```

## Reference Papers

- SafeDream: Safety World Model for Proactive Early Jailbreak Detection, arXiv:2604.16824
- Attacks Are All You Need to Break LLMs / IHO, arXiv:2606.03647
