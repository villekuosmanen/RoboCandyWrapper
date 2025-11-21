# v2.1 + v3.0 Mixed Dataset Loading

**Status: ⚠️ Infrastructure Complete - Needs Compatible Dataset Testing**

## What's Working

✅ **Infrastructure:**
- Both v2.1 and v3.0 load independently
- Automatic version detection
- Shape validation (catches incompatible robots)
- Weighted sampling on v2.1-only datasets

❓ **Needs Testing:**
- Mixed v2.1 + v3.0 loading with **compatible datasets** (same robot config)

## Directory Structure

### v2.1 (Per-Episode Files)
```
dataset/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/
│   ├── episode-0000.parquet
│   ├── episode-0001.parquet
│   └── ...
└── videos/chunk-000/
    ├── observation.images.*/
        ├── episode-0000.mp4
        └── ...
```

### v3.0 (Sharded Files)
```
dataset/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── file-0000.parquet
├── data/chunk-000/
│   ├── file-0000.parquet  # Many episodes
│   └── ...
└── videos/chunk-000/
    ├── observation.images.*/
        ├── file-0000.mp4  # Many episodes
        └── ...
```

## Requirements

**Datasets must have matching shapes** for common features:
- ❌ Cannot mix 6-DOF + 7-DOF robots
- ✅ Can mix v2.1 + v3.0 from **same robot configuration**

## Usage

```python
from robocandywrapper.factory import make_dataset_without_config

# Mix datasets (automatically detects v2.1 vs v3.0)
dataset = make_dataset_without_config([
    "username/dataset_v21",  # Old format
    "username/dataset_v30",  # New format
])

# Use with sampler config
config = {
    "dataset_weights": {
        "username/dataset_v21": 1.0,
        "username/dataset_v30": 2.0,
    }
}
```

## Implementation

- `LegacyLeRobotDataset` handles v2.1 format
- `legacy_utils.py` provides v2.1-specific utilities (self-contained, lerobot 0.4.1 compatible)
- `BackwardCompatibilityError` triggers automatic fallback to legacy loader
- Shape validation in `WrappedRobotDataset.__init__()` catches incompatible datasets early
