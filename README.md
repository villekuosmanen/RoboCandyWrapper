# 🍬 RoboCandyWrapper

**Sweet wrappers for extending and remixing LeRobot Datasets.**

---

## Why do I need this?

You have robot data. Lots of it. But it's messy:
*   Some datasets are from the "old days" (LeRobot v2.1).
*   Some are brand new (LeRobot v3.0).
*   Some are missing data you need for your brand new idea.

Traditionally, you'd have to write complex scripts to convert everything to a single format. **RoboCandyWrapper** handles that compatibility layer for you. It wraps your datasets in a sweet, consistent interface so you can focus on training, not data plumbing.

## Quick Start (5 Minutes)

### 1. Install
```bash
pip install robocandywrapper
```

### 2. The "Magic" Snippet
Load a vintage v2.1 dataset and a modern v3.0 dataset as if they were the same thing.

```python
from robocandywrapper import make_dataset_without_config

# Your playlist: one old, one new
repo_ids = [
    "lerobot/pusht_v2",      # v2.1 format
    "lerobot/aloha_v3",      # v3.0 format
]

# The factory handles the compatibility logic automatically
dataset = make_dataset_without_config(repo_ids)

print(f"🎉 Successfully loaded {len(dataset)} episodes from mixed sources!")
```

## What more can I do with it?

### 🎧 [The "Mix Tape" (Mixing Datasets)](docs/guide_mixing_datasets.md)
Learn how to combine multiple datasets into one, handle different robot configurations, and use sampling weights to balance your data mix.

### 🧂 [The "Flavor Enhancer" (Transforming Data)](docs/guide_transforming_data.md)
> **⚠️ Under Development:** Learn how to use **Plugins** to add new data fields, reshape tensors, or modify existing data on-the-fly without changing your original files. This feature is still under active development and may not be suitable for production use.


