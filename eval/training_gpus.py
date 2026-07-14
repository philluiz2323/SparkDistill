"""Accepted training-track GPU claims for proof bundles."""

from __future__ import annotations

import json

# Substrings matched case-insensitively against manifest train_gpu claims.
ACCEPTED_TRAINING_GPU_SUBSTRINGS = (
  # Blackwell workstation (RTX PRO 6000, etc.)
    "rtx pro 6000",
    "gb20",
    # Datacenter Blackwell
    "b200",
    "b300",
    "gb200",
    # Hopper training nodes
    "h100",
    "h200",
    "gh100",
    "gh200",
)


def is_accepted_training_gpu(train_gpu: str | None) -> bool:
    if not train_gpu:
        return False
    blob = str(train_gpu).lower()
    return any(pattern in blob for pattern in ACCEPTED_TRAINING_GPU_SUBSTRINGS)


def accepted_training_gpu_label() -> str:
    return "Blackwell (RTX PRO 6000 / B200 / B300) or Hopper (H100 / H200) CC node"


def attestation_corroborates_training_gpu(train_gpu: str, attestation: dict | None) -> bool:
    """When attestation is present, hwmodel claims must match the declared train_gpu family."""
    if not attestation:
        return True
    claims_blob = json.dumps(attestation.get("claims") or {}).lower()
    if claims_blob == "{}":
        return True

    gpu = str(train_gpu).lower()
    if any(token in gpu for token in ("h100", "gh100")):
        return "h100" in claims_blob or "gh100" in claims_blob
    if any(token in gpu for token in ("h200", "gh200")):
        return "h200" in claims_blob or "gh200" in claims_blob
    if any(token in gpu for token in ("b200", "b300", "gb200")):
        return any(token in claims_blob for token in ("b200", "b300", "gb200", "gb102"))
    if "pro 6000" in gpu or "gb20" in gpu:
        return "pro 6000" in claims_blob or "gb20" in claims_blob
    return is_accepted_training_gpu(train_gpu)
