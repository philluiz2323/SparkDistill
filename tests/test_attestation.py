import json

import pytest

from eval.attestation import _decode_overall_claims
from eval.verify import check_training_claims

jwt = pytest.importorskip("jwt")


def _token(overall: dict, devices: dict[str, dict]) -> str:
    encode = lambda payload: jwt.encode(payload, "k", algorithm="HS256")  # noqa: E731
    return json.dumps(
        [
            ["JWT", encode(overall)],
            {"REMOTE_GPU_CLAIMS": [["JWT", encode({"sub": "platform"})], {k: encode(v) for k, v in devices.items()}]},
        ]
    )


def test_decode_includes_device_hardware_claims():
    token = _token(
        {"iss": "NRAS", "x-nvidia-overall-att-result": True},
        {"GPU-0": {"hwmodel": "GB20X", "x-nvidia-gpu-driver-version": "595.71.05"}},
    )
    claims = _decode_overall_claims(token)
    assert claims["iss"] == "NRAS"
    assert claims["devices"]["GPU-0"]["hwmodel"] == "GB20X"


def test_device_claims_corroborate_training_gpu():
    # The overall JWT has no hardware fields; without device submodule claims the
    # verify-side corroboration check wrongly rejected genuinely attested bundles.
    token = _token({"iss": "NRAS"}, {"GPU-0": {"hwmodel": "GB20X"}})
    attestation = {"passed": True, "claims": _decode_overall_claims(token)}
    manifest = {"train_hours": 0.1, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    assert check_training_claims(manifest, attestation) == []


def test_garbage_token_decodes_to_empty():
    assert _decode_overall_claims("not json") == {}
