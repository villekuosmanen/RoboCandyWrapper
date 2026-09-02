import sys

import lerobot.datasets.utils as lerobot_dataset_utils
import pandas as pd
import pytest
import torch

# Current main imports these helpers from the newer module location. Alias the
# older location when running against the repository's declared LeRobot range.
sys.modules.setdefault("lerobot.datasets.feature_utils", lerobot_dataset_utils)
sys.modules.setdefault("lerobot.datasets.io_utils", lerobot_dataset_utils)

from robocandywrapper.wrapper import WrappedRobotDataset


class PandasTasks:
    def __init__(self, index_to_task: dict[int, str]):
        self._frame = pd.DataFrame(
            {"task_index": list(index_to_task)},
            index=pd.Index(index_to_task.values(), name="task"),
        )

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


class MockMetadata:
    def __init__(self, tasks):
        self.tasks = tasks
        self.stats = {}
        self.info = {"fps": 20, "video": False}
        self.camera_keys = []
        self.image_keys = []
        self.video_keys = []
        self.total_frames = 1
        self.total_episodes = 1
        self.episodes = {}

    @property
    def fps(self):
        return self.info["fps"]

    @property
    def features(self):
        return {
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }


class MockDataset:
    def __init__(self, repo_id: str, tasks, task_index):
        self.repo_id = repo_id
        self.meta = MockMetadata(tasks)
        self.features = self.meta.features
        self.hf_features = self.features
        self._task_index = task_index
        self.episodes = None
        self.num_episodes = 1

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "episode_index": torch.tensor(0),
            "task_index": (
                self._task_index.clone()
                if isinstance(self._task_index, torch.Tensor)
                else self._task_index
            ),
        }


def test_single_dataset_sample_uses_sorted_wrapper_task_index():
    dataset = MockDataset(
        "single",
        PandasTasks({0: "z task", 1: "a task"}),
        torch.tensor(0, dtype=torch.int64),
    )

    wrapped = WrappedRobotDataset(dataset)
    item = wrapped[0]

    assert wrapped.meta.tasks == {0: "a task", 1: "z task"}
    assert torch.equal(item["task_index"], torch.tensor(1, dtype=torch.int64))
    assert item["task"] == "z task"


def test_mixed_dataset_samples_remap_colliding_local_task_indices():
    first = MockDataset("first", {0: "z task"}, 0)
    second = MockDataset("second", PandasTasks({0: "a task"}), 0)

    wrapped = WrappedRobotDataset([first, second])

    assert wrapped.meta.tasks == {0: "a task", 1: "z task"}
    assert wrapped[0]["task_index"] == 1
    assert wrapped[0]["task"] == "z task"
    assert wrapped[1]["task_index"] == 0
    assert wrapped[1]["task"] == "a task"


def test_unknown_inner_task_index_has_contextual_error():
    dataset = MockDataset("broken", {0: "known task"}, 9)
    wrapped = WrappedRobotDataset(dataset)

    with pytest.raises(KeyError, match=r"task_index 9.*broken"):
        wrapped[0]


def test_real_busybox_sample_remaps_task_index_without_changing_task(monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    inner = LeRobotDataset(
        "villekuosmanen/busybox_multitask",
        episodes=[12],
        download_videos=False,
    )
    monkeypatch.setattr(inner, "_query_videos", lambda *args, **kwargs: {})

    inner_item = inner[0]
    inner_task_index = int(inner_item["task_index"])
    inner_task = str(inner.meta.tasks.index[inner.meta.tasks["task_index"] == inner_task_index][0])

    wrapped = WrappedRobotDataset(inner)
    wrapped_item = wrapped[0]
    wrapped_task_index = int(wrapped_item["task_index"])

    assert inner_task_index == 4
    assert inner_task == "Move the right slider to position 5"
    assert wrapped.meta.tasks[4] == "Move the left slider to position 5"
    assert wrapped_task_index == 9
    assert wrapped.meta.tasks[wrapped_task_index] == inner_task
    assert wrapped_item["task"] == inner_task
