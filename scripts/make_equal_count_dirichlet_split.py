#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from args import parse
from data import Data
from dataset_catalog import dataset_metadata_dict


class EqualCountDirichletData(Data):
    def split_dataset_by_dirichlet(self):
        train_labels = np.asarray(getattr(self.train_dataset, "targets"))
        total_samples = int(len(train_labels))
        if total_samples % int(self.num_clients) != 0:
            raise ValueError(
                f"Cannot make exactly equal client sample counts: "
                f"{total_samples} samples / {self.num_clients} clients is not an integer."
            )

        per_client = total_samples // int(self.num_clients)
        n_classes = int(train_labels.max()) + 1
        client_indices = [[] for _ in range(int(self.num_clients))]
        remaining_capacity = np.full(int(self.num_clients), per_client, dtype=np.int64)

        class_order = self.rng.permutation(n_classes)
        for class_id in class_order:
            indices = np.flatnonzero(train_labels == class_id)
            shuffled_indices = np.array(indices, copy=True)
            self.rng.shuffle(shuffled_indices)
            cursor = 0
            remaining_for_class = len(shuffled_indices)
            base_probs = self.rng.dirichlet([self.alpha] * int(self.num_clients))

            while remaining_for_class > 0:
                eligible = remaining_capacity > 0
                if not np.any(eligible):
                    raise RuntimeError("No client capacity left before all class samples were assigned.")

                probs = base_probs * eligible.astype(np.float64) * remaining_capacity.astype(np.float64)
                if float(probs.sum()) <= 0.0:
                    probs = eligible.astype(np.float64)
                probs = probs / probs.sum()

                alloc = self.rng.multinomial(remaining_for_class, probs)
                alloc = np.minimum(alloc, remaining_capacity)
                if int(alloc.sum()) == 0:
                    best_client = int(np.argmax(probs))
                    alloc[best_client] = min(remaining_for_class, int(remaining_capacity[best_client]))

                for client_id, count in enumerate(alloc):
                    count = int(count)
                    if count <= 0:
                        continue
                    chunk = shuffled_indices[cursor : cursor + count]
                    client_indices[client_id].append(chunk)
                    cursor += count
                    remaining_capacity[client_id] -= count
                    remaining_for_class -= count

        if not np.all(remaining_capacity == 0):
            raise RuntimeError(f"Unfilled client capacities after split: {remaining_capacity.tolist()}")

        client_subsets = []
        for chunks in client_indices:
            merged = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)
            self.rng.shuffle(merged)
            client_subsets.append(Subset(dataset=self.train_dataset, indices=merged))
        return client_subsets

    def _write_metadata(self):
        artifact_format = "dataset_references" if self._uses_reference_artifacts() else "materialized_samples"
        metadata = dataset_metadata_dict(
            self.data_name,
            num_clients=self.num_clients,
            alpha=self.alpha,
            seed=self.seed,
            artifact_format=artifact_format,
        )
        metadata.update(
            {
                "split_strategy": "equal_count_dirichlet_label_distribution",
                "equal_client_sample_count": True,
                "samples_per_client": len(self.train_dataset) // int(self.num_clients),
            }
        )
        metadata_path = Path(self.data_save_path) / "dataset_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    def _write_client_label_stats(self, client_subsets):
        stats = []
        for client_id, subset in enumerate(client_subsets, start=1):
            labels = [int(subset.dataset[idx][1]) for idx in subset.indices]
            counts = np.bincount(labels, minlength=int(self.num_classes)).astype(int)
            stats.append(
                {
                    "client_id": client_id,
                    "sample_count": int(len(labels)),
                    "num_classes_present": int(np.count_nonzero(counts)),
                    "class_counts": counts.tolist(),
                }
            )
        stats_path = Path(self.data_save_path) / "client_label_stats.json"
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)

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
        self._write_client_label_stats(client_subsets)


if __name__ == "__main__":
    EqualCountDirichletData(parse=parse).save_dataset()
