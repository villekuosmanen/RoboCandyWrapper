# Guide: The "Flavor Enhancer" (Transforming Data)

As a researcher, the models you train are only as good as our data. While the LeRobot data format already contains many common columns, sometimes we want to enrich it with extra information, either at episode or frame level. This might include:

- Subtask language instructions
- Episode outcome information
- Labelled affordance location or label
- Anything you want to include in your model training!

We can edit the LeRobotDataset class directly to include this data, but this creates two major issues:

1. Your Dataset code diverges from the LeRobot code in `main` - in order to use the new columns, users are forever tied to your data class.
2. The data loading and access code in LeRobotDataset doesn't always handle unexpected data well, and could cause issues. In the worst case, your dataset is in effect corrupted, and impossible to load using the base LeRobot class, potentially requiring migration scripts and complex data engineering to revert the changes in the future. I've seen issues like this happen when adding the following data:
    
    - Sparse data, that only exists for certain episodes or frames in a dataset.
    - Lists of strings, as some data processing code expects to convert them to tensors.

## The Solution: Plugins

Plugins hook the datasets at runtime, and enrich frames accessed via the `__getitem__` method of your dataset with additional columns. These columns can be loaded from the LeRobotDataset's repository if they are written to disk, or they can be runtime-only, either pre-calculated when lading the dataset, lazily initialised, or calculated on the fly.

### Example - Episode Outcome Plugin.

When evaluating policies, or when training them with reinforcement learning under sparse rewards, we need to know whether an episode was a success or not. This data can be included in LeRobotDatasets using the [Episode Outcome Plugin](robocandywrapper/plugins/episode_outcome.py).

We can label episode outcomes using the [labeling script](examples/label_episode_outcomes.py), which marks episodes as success or failure based on the file containing the labels (a human will have to provide this information).

We can then load the plugin during the creation of the dataset.
```
from robocandywrapper import make_dataset_without_config

dataset = make_dataset_without_config(repo_id, plugins=[EpisodeOutcomePlugin()])
```

We can then access the episode outcome as boolean tensors using the `episode_outcome` and `episode_outcome_mask` columns items loaded from our `DataLoader`. From the data consumer's perspective, they look the same as columns coming from `LeRobotDataset` directly.

### Storing data on disk

The Episode Outcome plugin stores the outcome labels in the same Hugging Face repository your LeRobot data is stored (or on disk, if you are not uploading your data to Hugging Face). This means you only need to label the episode outcomes once.

Most importantly, the labels are stored in the `candywrapper_plugins/episode_outcome` directory, fully isolated from the directories where LeRobot stores its data. This ensures modifications made by RoboCandyWrapper do not impact loading the dataset into "vanilla" LeRobot, preserving backwards compatibility.

**`candywrapper_plugins/<your_plugin>` is the recommended best practice location format for writing data into in custom Plugin classes, to ensure data is isolated (with best efforts) between different plugins.** This is just a convention though - the library doesn't enforce any particular location or data format.

### Mixing multiple plugins

The Plugin architecture is designed for composability - the framework allows loading in as many plugins as you want. This allows isolating data experiments from each other, allowing researchers to only load the plugins they need for each one.

Data is passed between Plugins using a *daisy chain* system. This means we can define plugins that depend on data from other plugins! For example, we can define `YourPlugin` that depends on the `episode_outcome` and `episode_outcome_mask` columns exposed by the `EpisodeOutcomePlugin`. To use this, simply ensure `EpisodeOutcomePlugin` is loaded first:

```
from robocandywrapper import make_dataset_without_config

dataset = make_dataset_without_config(repo_id, plugins=[EpisodeOutcomePlugin(), YourPlugin()])
```

Ordering of plugins matters - EpisodeOutcomePlugin needs to be loaded before YourPlugin for the daisy chaining of plugins to work. Flip the order and `YourPlugin` won't see the `episode_outcome` and `episode_outcome_mask` fields.
