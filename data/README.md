# Data Directory

This directory is intentionally kept out of Git because the generated files are large.

Expected local layout after setup:

```text
data/
  xguard-train/
    xguard-train.json
    transitions/
      xguard_transitions.jsonl
      xguard_transitions_wildguard.clean.jsonl
      xguard_transitions_wildguard_harmbench.clean.jsonl
      xguard_transitions_rewards.jsonl
    splits/
      train.jsonl
      val.jsonl
      test.jsonl
      split_stats.json
```

Use the commands in the project README to recreate these files on a server.
