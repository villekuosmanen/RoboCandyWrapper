# Guide: The "Mix Tape" (Mixing Datasets)

So you have a collection of datasets. Some are vintage v2.1 classics, others are fresh v3.0 releases. You want to train a model on *all* of them.

In the past, this meant converting everything to a single format. With RoboCandyWrapper, you can just **mix and match**.

## The Basics: Mixing v2.1 and v3.0

The `make_dataset_without_config` factory handles the compatibility logic for you. It automatically detects the version of each dataset and wraps it in the appropriate compatibility layer.

```python
from robocandywrapper import make_dataset_without_config

# Your playlist of datasets
repo_ids = [
    "lerobot/pusht_v2",      # v2.1 format dataset
    "lerobot/aloha_v3",      # v3.0 format dataset
]

# Create the mixed dataset
dataset = make_dataset_without_config(repo_ids)

print(f"Total episodes: {len(dataset)}")
# > Total episodes: 500 (200 from pusht + 300 from aloha)
```

## Advanced Mixing: Sampling Weights

You can control how often data is sampled from each dataset using sampling weights. A weight of 2.0 effectively doubles the size of that dataset in the mix (samples are drawn twice as often).

```python
config = {
    "dataset_weights": {
        "lerobot/pusht_v2": 1.0,  # Standard weight
        "lerobot/aloha_v3": 2.0,  # Effectively doubles this dataset's size in the mix
    }
}

dataset = make_dataset_without_config(repo_ids, config=config)
```

## The Rules of the Mix

While RoboCandyWrapper is powerful, it can't perform magic. There is one golden rule:

> **⚠️ The Golden Rule:** All datasets in a mix must have compatible **shapes** for their common features.

For example:
*   ✅ **OK:** Mixing two datasets where `observation.state` is `(6,)` (both are 6-DOF robots).
*   ❌ **Not OK:** Mixing a 6-DOF robot dataset with a 7-DOF robot dataset.

If you try to mix incompatible shapes, RoboCandyWrapper will raise a `ValueError` to let you know which datasets are clashing.

## Troubleshooting

### "Shape Mismatch Error"
This means one of your datasets has a different feature dimension than the others.
**Fix:** Check the `info.json` of your datasets. You may need to preprocess your data to align shapes, or use compatible datasets. **Plugins** (see [Transforming Data](./guide_transforming_data.md)) are still under development and may not be suitable for production use.

### "Version Not Detected"
RoboCandyWrapper checks for specific files to guess the version (e.g., `episodes.jsonl` for v2.1).
**Fix:** Ensure your local dataset cache hasn't been corrupted. Try deleting the cached dataset and re-downloading.
