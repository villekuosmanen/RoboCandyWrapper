# Guide: The "Flavor Enhancer" (Adding Data)

Sometimes your dataset is *almost* perfect, but it's missing something. Maybe you need to add a "language_instruction" field, or perhaps a "quality_score" for data filtering.

Traditionally, you'd have to rewrite the entire dataset. With RoboCandyWrapper, you can just **wrap it**.

## The Concept: Adapters

Adapters allow you to modify the data on-the-fly as it is loaded. The underlying data files remain unchanged, but the Adapter intercepts the data loading process to inject or modify fields before they reach your training code.

You can chain multiple adapters together:
`Dataset` -> `LanguageAdapter` -> `QualityScoreAdapter` -> `TrainingCode`

## Example: Adding Language Instructions

Let's say you have a dataset of a robot picking up a cup, but it has no text description. You want to inject the string "pick up the cup" into every frame.

### 1. Define the Adapter

Create a class that inherits from `Wrapper`.

```python
from robocandywrapper import Wrapper
import torch

class LanguageInstructionAdapter(Wrapper):
    def __init__(self, dataset, instruction: str):
        super().__init__(dataset)
        self.instruction = instruction

    def __getitem__(self, idx):
        # 1. Get the original data
        item = self.dataset[idx]
        
        # 2. Add our new flavor
        item["language_instruction"] = self.instruction
        
        return item
```

### 2. Wrap Your Dataset

Now you can apply this adapter to any dataset, regardless of its version.

```python
from robocandywrapper import make_dataset_without_config

# Load the base dataset
base_dataset = make_dataset_without_config(["lerobot/pusht_v2"])

# Wrap it!
enhanced_dataset = LanguageInstructionAdapter(base_dataset, "push the T-block")

# Verify
sample = enhanced_dataset[0]
print(sample["language_instruction"]) 
# > "push the T-block"
```

## Advanced: Modifying Existing Data

Adapters aren't just for adding new keys; they can also modify existing ones. This is useful for **fixing shape mismatches**.

For example, if you need to pad a 6-DOF action to 7-DOF to mix it with another dataset:

```python
class PadActionAdapter(Wrapper):
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Pad the 6th dimension to 7
        original_action = item["action"] # Shape (6,)
        padded_action = torch.cat([original_action, torch.zeros(1)])
        
        item["action"] = padded_action
        return item
```

## Best Practices

*   **Keep it Lightweight:** Adapters run on-the-fly. Avoid heavy computation in `__getitem__`.
*   **Chain Responsibly:** You can wrap a wrapper! `PadActionAdapter(LanguageInstructionAdapter(dataset))` works perfectly.
