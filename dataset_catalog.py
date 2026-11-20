from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    num_classes: int
    image_size: int
    in_channels: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    train_split: str
    test_split: str
    val_split: Optional[str] = None
    notes: Tuple[str, ...] = ()


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        name="cifar10",
        display_name="CIFAR-10",
        num_classes=10,
        image_size=32,
        in_channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        train_split="train",
        test_split="test",
        notes=(
            "Uses the official torchvision train/test split.",
            "Offline preprocessing is tensor conversion plus per-channel normalization only.",
        ),
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        display_name="CIFAR-100",
        num_classes=100,
        image_size=32,
        in_channels=3,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        train_split="train",
        test_split="test",
        notes=(
            "Uses the official torchvision train/test split.",
            "Offline preprocessing is tensor conversion plus per-channel normalization only.",
        ),
    ),
    "cinic10": DatasetSpec(
        name="cinic10",
        display_name="CINIC-10",
        num_classes=10,
        image_size=32,
        in_channels=3,
        mean=(0.4789, 0.4723, 0.4305),
        std=(0.2421, 0.2383, 0.2587),
        train_split="train",
        val_split="valid",
        test_split="test",
        notes=(
            "Expects the standard CINIC-10 folder layout with train/valid/test subdirectories.",
            "Server validation uses the official valid split and server test uses the official test split.",
        ),
    ),
    "tinyimagenet": DatasetSpec(
        name="tinyimagenet",
        display_name="Tiny-ImageNet",
        num_classes=200,
        image_size=64,
        in_channels=3,
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2302, 0.2265, 0.2262),
        train_split="train",
        test_split="val",
        notes=(
            "Uses the official Tiny-ImageNet train split for client partitioning.",
            "Uses the official Tiny-ImageNet val split as server_test because the public test split has no labels.",
            "Images stay at 64x64 and are not downsampled to 32x32.",
        ),
    ),
}


def get_dataset_spec(data_name: str) -> DatasetSpec:
    normalized_name = str(data_name).lower()
    if normalized_name not in DATASET_SPECS:
        raise KeyError(f"Unsupported dataset: {data_name}")
    return DATASET_SPECS[normalized_name]


def dataset_metadata_dict(
    data_name: str,
    *,
    num_clients: int,
    alpha: float,
    seed: int,
    artifact_format: str,
) -> Dict[str, object]:
    spec = get_dataset_spec(data_name)
    return {
        "data_name": spec.name,
        "display_name": spec.display_name,
        "num_classes": int(spec.num_classes),
        "image_size": int(spec.image_size),
        "in_channels": int(spec.in_channels),
        "normalization_mean": list(spec.mean),
        "normalization_std": list(spec.std),
        "train_split": spec.train_split,
        "val_split": spec.val_split,
        "test_split": spec.test_split,
        "num_clients": int(num_clients),
        "alpha": float(alpha),
        "seed": int(seed),
        "artifact_format": str(artifact_format),
        "offline_augmentation": "none",
        "notes": list(spec.notes),
    }


def format_dataset_notes(data_name: str) -> List[str]:
    return list(get_dataset_spec(data_name).notes)
