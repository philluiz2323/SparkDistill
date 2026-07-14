"""Frozen GSM8K regression sample for attested no-GPU verification.

Miners run `eval.export_gsm8k_regression_sample` on their checkpoint, include the
output in a proof bundle, and bind it with GPU attestation (`claim_sha256` nonce).
Validators re-grade the bundled responses against the frozen gold answers without
re-running the model when attestation + binding pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eval.benchmarks import BENCHMARKS
from eval.dataset_verify import _sha256_file

REGRESSION_VERSION = "sparkdistill-gsm8k-regression-v2"
REGRESSION_PROBLEM_COUNT = 50
REGRESSION_PROBLEMS_PATH = (
    Path(__file__).resolve().parent / "data" / f"gsm8k_regression_{REGRESSION_PROBLEM_COUNT}.jsonl"
)
REGRESSION_SAMPLE_FILENAME = "gsm8k_regression_sample.json"
REGRESSION_BENCHMARK_KEY = "gsm8k"


def load_regression_problems(path: Path = REGRESSION_PROBLEMS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "problem_id" not in row or "question" not in row or "answer" not in row:
                raise ValueError(f"{path}:{line_no}: each row needs problem_id, question, answer")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: regression problem set is empty")
    return rows


def regression_problem_set_sha256(path: Path = REGRESSION_PROBLEMS_PATH) -> str:
    return _sha256_file(path)


def normalize_gsm8k_answer(text: str) -> str:
    """Normalize a GSM8K final answer for exact-match grading."""
    value = str(text).strip()
    if "####" in value:
        value = value.split("####")[-1]
    value = value.replace("$", "").replace(",", "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def grade_response(gold: str, model_response: str) -> bool:
    return normalize_gsm8k_answer(gold) == normalize_gsm8k_answer(model_response)


def compute_exact_match(responses: list[dict[str, Any]], problems: list[dict[str, Any]]) -> float:
    by_id = {int(row["problem_id"]): row for row in problems}
    correct = 0
    for item in responses:
        problem_id = int(item["problem_id"])
        if problem_id not in by_id:
            raise ValueError(f"unknown problem_id {problem_id}")
        if grade_response(by_id[problem_id]["answer"], str(item["model_response"])):
            correct += 1
    return correct / len(problems)


def build_regression_sample(
    responses: list[dict[str, Any]],
    *,
    problems_path: Path = REGRESSION_PROBLEMS_PATH,
) -> dict[str, Any]:
    problems = load_regression_problems(problems_path)
    if len(responses) != len(problems):
        raise ValueError(f"expected {len(problems)} responses, got {len(responses)}")
    seen = {int(item["problem_id"]) for item in responses}
    expected = {int(row["problem_id"]) for row in problems}
    if seen != expected:
        raise ValueError(f"response problem_id set {sorted(seen)} != expected {sorted(expected)}")
    exact_match = compute_exact_match(responses, problems)
    return {
        "version": REGRESSION_VERSION,
        "benchmark": REGRESSION_BENCHMARK_KEY,
        "problem_set_path": problems_path.name,
        "problem_set_sha256": regression_problem_set_sha256(problems_path),
        "rows_total": len(problems),
        "exact_match": exact_match,
        "responses": sorted(responses, key=lambda row: int(row["problem_id"])),
    }


def verify_regression_sample(
    sample: dict[str, Any],
    *,
    claimed_gsm8k: float | None = None,
    problems_path: Path = REGRESSION_PROBLEMS_PATH,
    score_tolerance_pct: float = 0.5,
) -> list[str]:
    """CPU-only verification of a bundled GSM8K regression sample."""
    issues: list[str] = []
    if sample.get("version") != REGRESSION_VERSION:
        issues.append(f"regression sample version must be {REGRESSION_VERSION!r}")
    if sample.get("benchmark") != REGRESSION_BENCHMARK_KEY:
        issues.append(f"regression sample benchmark must be {REGRESSION_BENCHMARK_KEY!r}")

    expected_sha = regression_problem_set_sha256(problems_path)
    if sample.get("problem_set_sha256") != expected_sha:
        issues.append("regression sample problem_set_sha256 does not match frozen problem set")

    problems = load_regression_problems(problems_path)
    responses = sample.get("responses")
    if not isinstance(responses, list):
        return issues + ["regression sample missing responses list"]

    try:
        recomputed = compute_exact_match(responses, problems)
    except ValueError as exc:
        return issues + [str(exc)]

    reported = sample.get("exact_match")
    if not isinstance(reported, (int, float)):
        issues.append("regression sample missing exact_match score")
    elif abs(float(reported) - recomputed) > 1e-9:
        issues.append(
            f"regression sample exact_match {reported!r} does not match recomputed {recomputed!r}"
        )

    if claimed_gsm8k is not None and abs(float(claimed_gsm8k) - recomputed) * 100.0 > score_tolerance_pct:
        issues.append(
            f"claimed gsm8k {claimed_gsm8k!r} diverges from attested regression sample {recomputed!r}"
        )
    return issues


def check_gsm8k_no_regression(
    sample_score: float,
    frontier_gsm8k: float,
    *,
    floor_pct: float | None = None,
) -> list[str]:
    if floor_pct is None:
        floor_pct = BENCHMARKS[REGRESSION_BENCHMARK_KEY].regression_floor_pct
    if frontier_gsm8k == 0:
        return []
    pct_delta = (sample_score - frontier_gsm8k) / frontier_gsm8k * 100.0
    if pct_delta < 0 and abs(pct_delta) > floor_pct:
        return [f"gsm8k regression: {pct_delta:.2f}% vs frontier exceeds -{floor_pct}% floor"]
    return []


def read_regression_sample(bundle_dir: Path) -> dict[str, Any] | None:
    sample_path = bundle_dir / REGRESSION_SAMPLE_FILENAME
    if not sample_path.exists():
        return None
    return json.loads(sample_path.read_text(encoding="utf-8"))
