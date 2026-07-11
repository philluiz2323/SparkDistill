"""GPU confidential-computing attestation for proof-of-training submissions.

Wraps NVIDIA's `nv-attestation-sdk` (https://github.com/NVIDIA/nvtrust) so a miner's
proof-of-training claim can be backed by real evidence that the GPU it ran on was
inside a Confidential Computing (CC) trust boundary — e.g. a Blackwell RTX PRO 6000
Server Edition CC node — rather than just an unverifiable checkpoint + score claim.

Deprecation note: `nv-attestation-sdk` is EOL 2026-09-15. Its replacement
(https://github.com/NVIDIA/attestation-sdk) is C++/Rust-only today with no Python
bindings, so this remains the only real, currently-installable Python attestation
path. Install with `uv sync --extra proof` (needs a real NVIDIA CC-capable GPU +
driver stack to actually attest) — this module stays importable without it since the
SDK is only imported inside `attest_gpu`, not at module load time.

    python -m eval.attestation --out runs/<run-id>/attestation.json

Default remote-attestation endpoints below are NVIDIA's own public NRAS/RIM/OCSP
services, taken from `nv-attestation-sdk`'s own test fixtures (not guessed).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "gpu_remote_v3.json"

DEFAULT_NRAS_GPU_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
DEFAULT_RIM_URL = "https://rim.attestation.nvidia.com/v1/rim/"
DEFAULT_OCSP_URL = "https://ocsp.ndis.nvidia.com/"


@dataclass(frozen=True)
class AttestationResult:
    passed: bool
    environment: str
    token: str
    claims: dict


def attest_gpu(
    environment: str = "REMOTE",
    policy_path: Path = DEFAULT_POLICY_PATH,
    service_key: str | None = None,
    nras_gpu_url: str = DEFAULT_NRAS_GPU_URL,
    rim_url: str = DEFAULT_RIM_URL,
    ocsp_url: str = DEFAULT_OCSP_URL,
) -> AttestationResult:
    """Collect local GPU evidence and verify it (by default against NVIDIA's Remote
    Attestation Service — NRAS) against the appraisal policy at `policy_path`.

    Requires `nv-attestation-sdk` (`uv sync --extra proof`) and, for `environment="REMOTE"`,
    a GPU SKU that supports Confidential Computing (Hopper H100+, including the Blackwell
    RTX PRO 6000 Server Edition CC node this project targets) with a CC-enabled driver.
    """
    from nv_attestation_sdk import attestation as nv_attestation

    env = getattr(nv_attestation.Environment, environment.upper())

    client = nv_attestation.Attestation()
    client.set_name("sparkdistill-proof")
    if service_key is not None:
        client.set_service_key(service_key)
    client.add_verifier(nv_attestation.Devices.GPU, env, nras_gpu_url, "", ocsp_url=ocsp_url, rim_url=rim_url)

    evidence_list = client.get_evidence()
    passed = bool(client.attest(evidence_list))

    token = client.get_token() if passed else ""
    policy_text = Path(policy_path).read_text()
    validated = bool(client.validate_token(policy_text)) if passed else False

    claims = _decode_overall_claims(token) if token else {}
    nv_attestation.Attestation.reset()
    return AttestationResult(passed=passed and validated, environment=environment.upper(), token=token, claims=claims)


def _decode_overall_claims(token: str) -> dict:
    """Best-effort, unverified decode of the EAT's claims for display.

    Not the trust boundary — `validate_token` above (which checks signatures against
    the appraisal policy) is what actually decides `passed`.

    Besides the overall JWT, per-device submodule tokens are decoded under a
    `devices` key: they carry the hardware identity (`hwmodel`, driver/vbios
    versions) that `eval.verify.check_training_claims` corroborates the claimed
    training GPU against — the overall JWT alone has no hardware fields.
    """
    import jwt  # PyJWT, a transitive dep of nv-attestation-sdk

    def _decode(encoded: str) -> dict:
        return jwt.decode(encoded, options={"verify_signature": False})

    try:
        parsed = json.loads(token)
        claims = _decode(parsed[0][1])
        devices: dict = {}
        for section in parsed[1:]:
            if not isinstance(section, dict):
                continue
            # e.g. {"REMOTE_GPU_CLAIMS": [["JWT", <platform>], {"GPU-0": <jwt>, ...}]}
            for entries in section.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    for device, device_jwt in entry.items():
                        try:
                            devices[device] = _decode(device_jwt)
                        except Exception:
                            continue
        if devices:
            claims["devices"] = devices
        return claims
    except Exception:
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--environment", default="REMOTE", choices=["LOCAL", "REMOTE", "TEST"])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--service-key", default=None, help="NGC service key with NRAS access (optional)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = attest_gpu(environment=args.environment, policy_path=args.policy, service_key=args.service_key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"passed": result.passed, "environment": result.environment, "token": result.token, "claims": result.claims},
            indent=2,
        )
    )
    print(f"attestation {'PASSED' if result.passed else 'FAILED'} ({result.environment})", file=sys.stderr)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
