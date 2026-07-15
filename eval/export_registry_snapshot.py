"""Export the accepted-registry fingerprint snapshot miners pass to SparkProof.

SparkProof's release gate already accepts ``--registry-snapshot`` (JSONL of prior
trajectory rows). SparkDistill builds that file by simulating the same cross-registry
mix dedupe used for ``sparkproof-mining``, so miners can see registry duplicates
*before* opening a dataset PR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from eval.mining_dataset import DEFAULT_MINING_DATASET_REPO, mining_dedupe_mode
from eval.mix_registry import (
    DedupeMode,
    _add_row_to_registry,
    _classify_row,
    _make_dedupe_registry,
    _should_skip,
    load_trajectories_jsonl,
    resolve_proof_dir,
)

ACCEPTED_REGISTRY_SNAPSHOT_PATH = Path("datasets/accepted_registry_snapshot.jsonl")
ACCEPTED_TASK_IDS_PATH = Path("datasets/accepted_task_ids.json")
HF_SNAPSHOT_FILENAME = "accepted_registry_snapshot.jsonl"
HF_TASK_IDS_FILENAME = "accepted_task_ids.json"


def _task_id_from_trajectory(trajectory: dict[str, Any]) -> str | None:
    meta = trajectory.get("metadata") or {}
    prompt_meta = meta.get("prompt_meta") or {}
    value = prompt_meta.get("task_id") or prompt_meta.get("problem_id") or meta.get("task_id")
    return str(value) if value else None


def collect_accepted_trajectories(
    registry_entries: list[dict[str, Any]],
    *,
    sparkproof_root: Path | None = None,
    dedupe: DedupeMode | str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
    proof_cache: Path | None = None,
) -> list[dict[str, Any]]:
    """Return trajectory rows whose fingerprints occupy the accepted registry mix state."""
    mode: DedupeMode = (dedupe or mining_dedupe_mode())  # type: ignore[assignment]
    working_registry, fingerprint_row = _make_dedupe_registry(sparkproof_root)
    working = working_registry.copy() if hasattr(working_registry, "copy") else working_registry

    accepted: list[dict[str, Any]] = []
    for entry in registry_entries:
        proof_dir = resolve_proof_dir(entry, proof_cache=proof_cache, download_proof=download_proof)
        for trajectory in load_trajectories_jsonl(proof_dir / "trajectories.jsonl"):
            verdict = _classify_row(trajectory, working, dedupe=mode, fingerprint_row=fingerprint_row)
            if _should_skip(verdict, dedupe=mode):
                continue
            accepted.append(trajectory)
            _add_row_to_registry(working, trajectory, fingerprint_row)
    return accepted


def write_registry_snapshot(
    registry_entries: list[dict[str, Any]],
    *,
    out_path: Path = ACCEPTED_REGISTRY_SNAPSHOT_PATH,
    task_ids_path: Path = ACCEPTED_TASK_IDS_PATH,
    sparkproof_root: Path | None = None,
    dedupe: DedupeMode | str | None = None,
    download_proof: Callable[[str, Path | None], Path] | None = None,
) -> dict[str, Any]:
    """Write SparkProof-compatible snapshot JSONL plus a lightweight task-id index."""
    accepted = collect_accepted_trajectories(
        registry_entries,
        sparkproof_root=sparkproof_root,
        dedupe=dedupe,
        download_proof=download_proof,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    task_ids = sorted({task_id for row in accepted if (task_id := _task_id_from_trajectory(row))})
    payload = {
        "rows_total": len(accepted),
        "task_ids_total": len(task_ids),
        "dedupe_mode": dedupe or mining_dedupe_mode(),
        "task_ids": task_ids,
    }
    task_ids_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "snapshot_path": out_path,
        "task_ids_path": task_ids_path,
        "rows_total": len(accepted),
        "task_ids_total": len(task_ids),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }


def publish_registry_snapshot(
    snapshot_path: Path,
    *,
    repo_id: str = DEFAULT_MINING_DATASET_REPO,
    task_ids_path: Path | None = None,
) -> dict[str, Any]:
    """Upload snapshot artifacts beside mix_manifest.json on the mining HF repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(snapshot_path),
        path_in_repo=HF_SNAPSHOT_FILENAME,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update accepted registry snapshot for miner novelty checks",
    )
    if task_ids_path is not None and task_ids_path.exists():
        api.upload_file(
            path_or_fileobj=str(task_ids_path),
            path_in_repo=HF_TASK_IDS_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update accepted registry task-id index",
        )
    return {
        "published": True,
        "hf_url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{HF_SNAPSHOT_FILENAME}",
        "repo_id": repo_id,
        "rows_total": sum(1 for line in snapshot_path.read_text().splitlines() if line.strip()),
    }


def main(argv: list[str] | None = None) -> int:
    from eval.mix_registry import load_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("datasets/registry.jsonl"),
        help="registry file to export (default: datasets/registry.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ACCEPTED_REGISTRY_SNAPSHOT_PATH,
        help="local snapshot JSONL for SparkProof --registry-snapshot",
    )
    parser.add_argument(
        "--task-ids-out",
        type=Path,
        default=ACCEPTED_TASK_IDS_PATH,
        help="local JSON index of accepted task_ids",
    )
    parser.add_argument("--sparkproof-root", type=Path, default=None)
    parser.add_argument("--publish", action="store_true", help="upload snapshot to sparkproof-mining HF repo")
    parser.add_argument("--repo-id", default=DEFAULT_MINING_DATASET_REPO)
    args = parser.parse_args(argv)

    try:
        report = write_registry_snapshot(
            load_registry(args.registry),
            out_path=args.out,
            task_ids_path=args.task_ids_out,
            sparkproof_root=args.sparkproof_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"export registry snapshot failed: {exc}", file=sys.stderr)
        return 1

    if args.publish:
        try:
            report.update(
                publish_registry_snapshot(
                    args.out,
                    repo_id=args.repo_id,
                    task_ids_path=args.task_ids_out,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"publish registry snapshot failed: {exc}", file=sys.stderr)
            return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
