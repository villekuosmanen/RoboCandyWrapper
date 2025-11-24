# Sampler Configuration Examples

This directory contains example configuration files for the WeightedSampler.

## Files

### `sampler_config.json`

Main example configuration file with comments explaining each field.

**Usage:**
```bash
export SAMPLER_CONFIG_PATH=/path/to/RoboCandyWrapper/examples/configs/sampler_config.json

python examples/train.py --config-path=examples/configs/your_train_config.yaml
```

**What it does:**
- Oversamples `lerobot/svla_so100_stacking` datasets 3x
- Uses deterministic sampling with seed 42

## Configuration Fields

### Required Fields

- **`type`** (string): Sampler type. Currently only `"weighted"` is supported.

### Optional Fields

- **`dataset_weights`** (object): Maps dataset repo IDs to weight multipliers
  - Weights > 1.0: oversample
  - Weights < 1.0: undersample
  - Unlisted datasets get weight 1.0
  - **Default:** `null` (uniform sampling)

- **`episodes`** (object or null): Maps dataset repo IDs to list of episode indices
  - Filters datasets to only include specified episodes
  - Useful for train/val splits
  - **Default:** `null` (load all episodes)

- **`samples_per_epoch`** (integer or null): Total samples per epoch
  - **Default:** `null` (uses sum of dataset lengths)

- **`shuffle`** (boolean): Shuffle combined samples
  - **Default:** `true`

- **`seed`** (integer or null): Random seed for reproducibility
  - **Default:** `null` (non-deterministic)

- **`replacement`** (boolean): Sample with replacement
  - **Default:** `true`

## Common Use Cases

### Oversample Small Dataset

```json
{
  "type": "weighted",
  "dataset_weights": {
    "small_dataset": 5.0,
    "large_dataset": 1.0
  },
  "seed": 42,
  "replacement": true
}
```

### Equal Representation

```json
{
  "type": "weighted",
  "dataset_weights": {
    "dataset_a": 1.0,
    "dataset_b": 1.0,
    "dataset_c": 1.0
  },
  "samples_per_epoch": 30000,
  "seed": 42
}
```

### Emphasize Recent Data

```json
{
  "type": "weighted",
  "dataset_weights": {
    "old_dataset_v1": 0.3,
    "old_dataset_v2": 0.5,
    "new_dataset_v3": 2.0
  },
  "seed": 42
}
```

### Episode Selection (Train/Val Split)

```json
{
  "type": "weighted",
  "dataset_weights": {
    "dataset_a": 1.0
  },
  "episodes": {
    "dataset_a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  }
}
```

## Validation

Your JSON config will be validated when loaded. Common errors:

- **Invalid JSON syntax**: Use `python -m json.tool your_config.json` to check
- **Negative weights**: All weights must be positive numbers
- **Wrong types**: `shuffle` and `replacement` must be boolean
- **Invalid seed**: Must be an integer or null

## See Also

- [Training Script](../train.py)

