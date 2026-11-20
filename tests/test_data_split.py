import sys

import torch

from args import parse
from data import Data


class FakeClassificationDataset:
    def __init__(self, start: int, size: int, num_classes: int):
        self.start = start
        self.size = size
        self.targets = [idx % num_classes for idx in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        value = float(self.start + index)
        label = self.targets[index]
        return torch.tensor([value], dtype=torch.float32), label


def _load_saved_indices(path):
    samples = torch.load(path)
    return [int(sample[0].item()) for sample in samples]


def _build_data(monkeypatch, tmp_path, seed):
    monkeypatch.setattr(
        Data,
        "load_dataset",
        lambda self: (
            FakeClassificationDataset(start=0, size=24, num_classes=4),
            FakeClassificationDataset(start=100, size=8, num_classes=4),
            4,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--data_name",
            "cifar10",
            "--num_clients",
            "4",
            "--alpha",
            "0.5",
            "--seed",
            str(seed),
            "--data_save_path",
            str(tmp_path),
        ],
    )
    return Data(parse=parse)


def test_cifar_save_dataset_only_writes_client_train_and_server_test(monkeypatch, tmp_path):
    data = _build_data(monkeypatch, tmp_path, seed=7)
    data.save_dataset()

    filenames = sorted(path.name for path in tmp_path.iterdir())
    assert filenames == [
        "1_train.pth",
        "2_train.pth",
        "3_train.pth",
        "4_train.pth",
        "server_test.pth",
    ]

    client_indices = []
    for client_id in range(1, 5):
        client_indices.extend(_load_saved_indices(tmp_path / f"{client_id}_train.pth"))
    server_test_indices = _load_saved_indices(tmp_path / "server_test.pth")

    assert len(client_indices) == 24
    assert len(server_test_indices) == 8
    assert all(index < 100 for index in client_indices)
    assert all(index >= 100 for index in server_test_indices)


def test_cifar_save_dataset_removes_legacy_split_files(monkeypatch, tmp_path):
    legacy_files = [
        "1_val.pth",
        "1_test.pth",
        "server_train.pth",
        "server_val.pth",
    ]
    for filename in legacy_files:
        (tmp_path / filename).write_text("legacy", encoding="utf-8")

    data = _build_data(monkeypatch, tmp_path, seed=11)
    data.save_dataset()

    for filename in legacy_files:
        assert not (tmp_path / filename).exists()


def test_cifar_dirichlet_split_is_reproducible_with_same_seed(monkeypatch, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_data = _build_data(monkeypatch, first_dir, seed=19)
    first_data.save_dataset()

    second_data = _build_data(monkeypatch, second_dir, seed=19)
    second_data.save_dataset()

    for client_id in range(1, 5):
        assert _load_saved_indices(first_dir / f"{client_id}_train.pth") == _load_saved_indices(
            second_dir / f"{client_id}_train.pth"
        )
    assert _load_saved_indices(first_dir / "server_test.pth") == _load_saved_indices(
        second_dir / "server_test.pth"
    )
