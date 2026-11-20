import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from args import parse
from dataset_catalog import dataset_metadata_dict, get_dataset_spec


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: str, class_to_idx, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = dict(class_to_idx)
        self.samples = self._build_samples()
        self.targets = [label for _, label in self.samples]

    def _build_samples(self):
        annotations_path = self.root / "val_annotations.txt"
        images_root = self.root / "images"
        if not annotations_path.exists() or not images_root.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet validation split is incomplete under {self.root}. "
                "Expected val_annotations.txt and images/."
            )

        samples = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("	")
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                if class_name not in self.class_to_idx:
                    continue
                image_path = images_root / image_name
                samples.append((str(image_path), int(self.class_to_idx[class_name])))
        if not samples:
            raise ValueError(f"No labeled Tiny-ImageNet validation samples found under {self.root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _resolve_existing_directory(candidates, expected_children):
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            continue
        if all((candidate_path / child).exists() for child in expected_children):
            return candidate_path
    searched = ", ".join(str(Path(candidate)) for candidate in candidates)
    expected = ", ".join(expected_children)
    raise FileNotFoundError(f"Could not find a dataset directory containing {expected}. Searched: {searched}")


def _resolve_tinyimagenet_root(data_path: str) -> Path:
    return _resolve_existing_directory(
        candidates=[
            Path(data_path) / "tiny-imagenet-200",
            Path(data_path) / "tiny-imagenet-200" / "tiny-imagenet-200",
            Path(data_path),
        ],
        expected_children=("train", "val", "wnids.txt"),
    )


def _resolve_cinic10_root(data_path: str) -> Path:
    return _resolve_existing_directory(
        candidates=[
            Path(data_path) / "cinic-10",
            Path(data_path) / "CINIC-10",
            Path(data_path) / "cinic10",
            Path(data_path),
        ],
        expected_children=("train", "valid", "test"),
    )


def _build_transform(data_name: str):
    spec = get_dataset_spec(data_name)
    transform_steps = []
    if int(spec.image_size) != 32:
        transform_steps.append(transforms.Resize((spec.image_size, spec.image_size)))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=spec.mean, std=spec.std),
        ]
    )
    return transforms.Compose(transform_steps)


class Data:
    def __init__(self, parse: argparse.ArgumentParser):
        self.args = parse.parse_args()
        self.data_save_path = self.args.data_save_path
        self.data_name = str(self.args.data_name).lower()
        self.num_clients = self.args.num_clients
        self.alpha = self.args.alpha
        self.seed = self.args.seed
        self.spec = get_dataset_spec(self.data_name)
        loaded = self.load_dataset()
        if len(loaded) == 3:
            self.train_dataset, self.test_dataset, self.num_classes = loaded
            self.server_val_dataset = None
        elif len(loaded) == 4:
            self.train_dataset, self.server_val_dataset, self.test_dataset, self.num_classes = loaded
        else:
            raise ValueError("load_dataset() must return either 3 or 4 values")
        self.rng = np.random.RandomState(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_dataset(self):
        transform = _build_transform(self.data_name)

        if self.data_name == "cifar10":
            train_dataset = CIFAR10(root=self.args.data_path, train=True, download=True, transform=transform)
            test_dataset = CIFAR10(root=self.args.data_path, train=False, download=True, transform=transform)
            return train_dataset, test_dataset, self.spec.num_classes

        if self.data_name == "cifar100":
            train_dataset = CIFAR100(root=self.args.data_path, train=True, download=True, transform=transform)
            test_dataset = CIFAR100(root=self.args.data_path, train=False, download=True, transform=transform)
            return train_dataset, test_dataset, self.spec.num_classes

        if self.data_name == "cinic10":
            cinic_root = _resolve_cinic10_root(self.args.data_path)
            train_dataset = ImageFolder(root=str(cinic_root / "train"), transform=transform)
            val_dataset = ImageFolder(root=str(cinic_root / "valid"), transform=transform)
            test_dataset = ImageFolder(root=str(cinic_root / "test"), transform=transform)
            return train_dataset, val_dataset, test_dataset, self.spec.num_classes

        if self.data_name == "tinyimagenet":
            tiny_root = _resolve_tinyimagenet_root(self.args.data_path)
            train_dataset = ImageFolder(root=str(tiny_root / "train"), transform=transform)
            test_dataset = TinyImageNetValDataset(
                root=str(tiny_root / "val"),
                class_to_idx=train_dataset.class_to_idx,
                transform=transform,
            )
            return train_dataset, test_dataset, self.spec.num_classes

        raise ValueError(f"Unsupported dataset: {self.data_name}")

    def split_dataset_by_dirichlet(self):
        train_labels = np.asarray(getattr(self.train_dataset, "targets"))
        n_classes = int(train_labels.max()) + 1
        label_distribution = self.rng.dirichlet([self.alpha] * self.num_clients, n_classes)
        class_indices = [np.flatnonzero(train_labels == class_id) for class_id in range(n_classes)]
        client_indices = [[] for _ in range(self.num_clients)]

        for indices, fractions in zip(class_indices, label_distribution):
            shuffled_indices = np.array(indices, copy=True)
            self.rng.shuffle(shuffled_indices)
            split_points = (np.cumsum(fractions)[:-1] * len(shuffled_indices)).astype(int)
            for client_idx, chunk in enumerate(np.split(shuffled_indices, split_points)):
                if len(chunk) > 0:
                    client_indices[client_idx].append(chunk)

        client_subsets = []
        for per_client_chunks in client_indices:
            if per_client_chunks:
                merged_indices = np.concatenate(per_client_chunks)
            else:
                merged_indices = np.empty((0,), dtype=np.int64)
            client_subsets.append(Subset(dataset=self.train_dataset, indices=merged_indices))
        return client_subsets

    def extract_data(self, subset: Subset):
        dataset = subset.dataset
        indices = subset.indices
        return [dataset[idx] for idx in indices]

    def extract_dataset(self, dataset: Dataset):
        return [dataset[idx] for idx in range(len(dataset))]

    def cleanup_saved_dataset(self):
        if not os.path.isdir(self.data_save_path):
            return

        legacy_pattern = re.compile(r"^\d+_(train|val|test)\.pth$")
        removable_names = {
            "server_train.pth",
            "server_val.pth",
            "server_test.pth",
            "dataset_metadata.json",
        }
        for filename in os.listdir(self.data_save_path):
            if filename in removable_names or legacy_pattern.match(filename):
                os.remove(os.path.join(self.data_save_path, filename))

    def _uses_reference_artifacts(self):
        return self.data_name in {"cinic10", "tinyimagenet"}

    def _serialize_client_train_split(self, client_subset: Subset):
        if self._uses_reference_artifacts():
            return client_subset
        return self.extract_data(client_subset)

    def _serialize_server_dataset(self, dataset: Dataset):
        if self._uses_reference_artifacts():
            return dataset
        return self.extract_dataset(dataset)

    def _write_metadata(self):
        artifact_format = "dataset_references" if self._uses_reference_artifacts() else "materialized_samples"
        metadata = dataset_metadata_dict(
            self.data_name,
            num_clients=self.num_clients,
            alpha=self.alpha,
            seed=self.seed,
            artifact_format=artifact_format,
        )
        metadata_path = Path(self.data_save_path) / "dataset_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    def save_dataset(self):
        os.makedirs(self.data_save_path, exist_ok=True)
        self.cleanup_saved_dataset()
        client_subsets = self.split_dataset_by_dirichlet()
        for client_id, client_subset in enumerate(client_subsets, start=1):
            payload = self._serialize_client_train_split(client_subset)
            torch.save(payload, os.path.join(self.data_save_path, f"{client_id}_train.pth"))

        if self.server_val_dataset is not None:
            server_val_payload = self._serialize_server_dataset(self.server_val_dataset)
            torch.save(server_val_payload, os.path.join(self.data_save_path, "server_val.pth"))

        server_test_payload = self._serialize_server_dataset(self.test_dataset)
        torch.save(server_test_payload, os.path.join(self.data_save_path, "server_test.pth"))
        self._write_metadata()


class TinyImageNetData(Data):
    pass


if __name__ == "__main__":
    Data(parse=parse).save_dataset()
