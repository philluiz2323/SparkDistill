"""Tests for eval.training_gpus."""

from eval.training_gpus import attestation_corroborates_training_gpu, is_accepted_training_gpu


def test_accepted_training_gpus():
    assert is_accepted_training_gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert is_accepted_training_gpu("NVIDIA B200")
    assert is_accepted_training_gpu("NVIDIA B300")
    assert is_accepted_training_gpu("NVIDIA H100 SXM")
    assert is_accepted_training_gpu("NVIDIA H200")
    assert not is_accepted_training_gpu("NVIDIA A100")


def test_attestation_corroboration_by_family():
    assert attestation_corroborates_training_gpu(
        "NVIDIA H100",
        {"passed": True, "claims": {"hwmodel": "NVIDIA GH100"}},
    )
    assert not attestation_corroborates_training_gpu(
        "NVIDIA H100",
        {"passed": True, "claims": {"hwmodel": "GB202 RTX PRO 6000"}},
    )
