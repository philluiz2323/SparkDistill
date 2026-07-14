"""CLI to refresh datasets/canonical.json from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.canonical_dataset import CANONICAL_PATH, write_pin_from_remote


def refresh_canonical_pin_file(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
    out_path: Path = CANONICAL_PATH,
) -> dict:
    """Download HF mix_manifest and rewrite datasets/canonical.json."""
    return write_pin_from_remote(repo_id=repo_id, hf_token=hf_token, out_path=out_path)


def commit_canonical_pin_to_main(
    *,
    repo_id: str | None = None,
    hf_token: str | None = None,
    out_path: Path = CANONICAL_PATH,
) -> list[str]:
    """Refresh the pin on the current branch and push to origin/main."""
    refresh_canonical_pin_file(repo_id=repo_id, hf_token=hf_token, out_path=out_path)
    if subprocess.run(["git", "diff", "--quiet", out_path.as_posix()]).returncode == 0:
        return []

    subprocess.run(["git", "add", out_path.as_posix()], check=True)
    commit = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Refresh canonical mining dataset pin after registry update.\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        return [commit.stderr.strip() or commit.stdout.strip() or "git commit failed"]

    push = subprocess.run(
        ["git", "push", "origin", "HEAD:main"],
        capture_output=True,
        text=True,
        check=False,
    )
    if push.returncode != 0:
        return [push.stderr.strip() or push.stdout.strip() or "git push failed"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=CANONICAL_PATH,
        help="path to write datasets/canonical.json",
    )
    parser.add_argument("--repo-id", default=None, help="HF datasets repo (default: canonical default)")
    args = parser.parse_args(argv)

    try:
        payload = refresh_canonical_pin_file(
            repo_id=args.repo_id,
            hf_token=os.environ.get("HF_TOKEN"),
            out_path=args.out,
        )
    except Exception as exc:
        print(f"update canonical pin failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
