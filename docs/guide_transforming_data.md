# Guide: The "Flavor Enhancer" (Transforming Data)

Sometimes your dataset is *almost* perfect, but it needs some tweaks. Maybe you need to add a "language_instruction" field, reshape tensors to match other datasets, or modify existing data on-the-fly.

Traditionally, you'd have to rewrite the entire dataset. With RoboCandyWrapper, you can use **plugins** to transform your data as it's loaded.

## The Concept: Plugins

Plugins allow you to modify the data on-the-fly as it is loaded. The underlying data files remain unchanged, but the Plugin intercepts the data loading process to inject or modify fields before they reach your training code.

You can use multiple plugins together - they all operate on the original dataset item and their outputs are merged into the final result.

## Example: Adding Language Instructions

Let's say you have a dataset of a robot picking up a cup, but it has no text description. You want to inject the string "pick up the cup" into every frame.

### 1. Define the Plugin

Create a class that inherits from `DatasetPlugin`.

```python
from robocandywrapper import DatasetPlugin, PluginInstance
import torch

class LanguageInstructionPlugin(DatasetPlugin):
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

### 2. Apply the Plugin

Now you can apply this plugin to any dataset, regardless of its version.

```python
from robocandywrapper import make_dataset_without_config

# Load the base dataset with the plugin
dataset = make_dataset_without_config(
    ["lerobot/svla_so100_pickplace"],
    plugins=[LanguageInstructionPlugin("pick up the red block")]
)

# Verify
sample = enhanced_dataset[0]
print(sample["language_instruction"]) 
# > "pick up the red block"
```

## Advanced: Modifying Existing Data

Plugins aren't just for adding new keys; they can also modify existing ones. This is useful for **fixing shape mismatches**.

For example, if you need to pad a 6-DOF action to 7-DOF to mix it with another dataset:

```python
class PadActionPlugin(DatasetPlugin):
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Pad the 6th dimension to 7
        original_action = item["action"] # Shape (6,)
        padded_action = torch.cat([original_action, torch.zeros(1)])
        
        item["action"] = padded_action
        return item
```

## Best Practices

*   **Keep it Lightweight:** Plugins run on-the-fly. Avoid heavy computation in `get_item_data`.
*   **Multiple Plugins:** You can apply multiple plugins together: `[PadActionPlugin(), LanguageInstructionPlugin()]`. They all operate on the original dataset item and their outputs are merged.
