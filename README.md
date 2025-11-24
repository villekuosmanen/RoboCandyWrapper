# 🍬 RoboCandyWrapper

**Sweet wrappers for extending and remixing LeRobot Datasets.**

---

## Why do I need this?

You have robot data. Lots of it. But it's messy:
*   Some datasets are from the "old days" (v2.1 dataset).
*   Some are brand new (v3.0 dataset).
*   You want to mix various data sources as needed, without permanently merging them.

Traditionally, you'd have to write complex scripts to convert everything to a single format. **RoboCandyWrapper** handles that compatibility layer for you. It wraps your datasets in a sweet, consistent interface so you can focus on training, not data plumbing.

Additionally, you might want to extend your datasets with additional labels and columns without breaking backwards compatibility of data or code with LeRobot. RoboCandyWrapper provides an extendible **Adapter** system to add new data to existing datasets, load any number of adapters during training, and mixing data between adapters.

RoboCandyWrapper also includes a **Sampler** system to change the ratio of sampling between multiple data sources, so you can increase or decrease the weight of specific datasets in your data mix as needed.

## Quick Start (5 Minutes)

### Installation
```bash
# Include LeRobot as a dependency in installation
pip install robocandywrapper

# OR...
# Use your own version of LeRobot - may cause issues!
pip install --no-dependencies robocandywrapper

# OR...
# Use your own version of LeRobot and install robocandywrapper as a local editable dependency so you change LeRobot imports as needed
# This might be required if you use a LeRobot fork or depend on an out of date version
git clone https://github.com/villekuosmanen/RoboCandyWrapper.git
cd RoboCandyWrapper
pip install --no-dependencies -e .
```

### Basic usage
Load a vintage v2.1 dataset and a modern v3.0 dataset as if they were the same thing.

```python
from robocandywrapper import make_dataset_without_config

# Your playlist: one old, one new
repo_ids = [
    "lerobot/svla_so100_pickplace",  # v2.1 dataset
    "lerobot/svla_so100_stacking",   # v3.0 dataset
]

# The factory handles the compatibility logic automatically
dataset = make_dataset_without_config(repo_ids)

print(f"🎉 Successfully loaded {len(dataset)} episodes from mixed sources!")
```

## What more can I do with it?

### 🎧 [The "Mix Tape" (Mixing Datasets)](docs/guide_mixing_datasets.md)
Learn how to combine multiple datasets into one, handle different robot configurations, and use sampling weights to balance your data mix.

### 🧂 [The "Flavor Enhancer" (Transforming Data)](docs/guide_transforming_data.md)
Learn how to use **Plugins** to add new labels or columns to your dataset, reshape tensors, or modify existing data on-the-fly without breaking backwards compatability.
