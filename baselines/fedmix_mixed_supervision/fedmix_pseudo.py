from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def generate_mask(student_probs: torch.Tensor, teacher_probs: torch.Tensor, threshold: float) -> Dict[str, torch.Tensor]:
    student_pred = student_probs.argmax(dim=1)
    teacher_pred = teacher_probs.argmax(dim=1)
    mean_probs = 0.5 * (student_probs + teacher_probs)
    confidence, pseudo_labels = mean_probs.max(dim=1)
    agreement = student_pred.eq(teacher_pred)
    mask = agreement & confidence.ge(float(threshold))
    consistency = (student_probs * teacher_probs).sum(dim=1)
    return {
        "mask": mask,
        "pseudo_labels": pseudo_labels,
        "pseudo_probs": mean_probs,
        "confidence": confidence,
        "consistency": consistency,
        "agreement": agreement,
    }


def generate_pseudo_labels(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    threshold: float,
    label_type: str = "class",
) -> Dict[str, torch.Tensor]:
    normalized_label_type = str(label_type).lower()
    if normalized_label_type not in {"class", "image", "unlabeled"}:
        raise ValueError(
            "This FedMix baseline adapts pseudo-labeling to classification-style labels only; "
            f"unsupported label_type={label_type!r}"
        )

    student_probs = F.softmax(student_logits.detach(), dim=1)
    teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
    return generate_mask(student_probs=student_probs, teacher_probs=teacher_probs, threshold=threshold)
