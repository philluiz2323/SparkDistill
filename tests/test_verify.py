from eval.verify import check_claim


def test_check_claim_within_tolerance_has_no_mismatch():
    claimed = {"gsm8k": 0.88, "humaneval": 0.80}
    rerun = {"gsm8k": 0.885, "humaneval": 0.795}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []


def test_check_claim_beyond_tolerance_flags_mismatch():
    claimed = {"gsm8k": 0.88, "humaneval": 0.80}
    rerun = {"gsm8k": 0.70, "humaneval": 0.795}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == ["gsm8k"]


def test_check_claim_ignores_benchmarks_not_claimed():
    claimed = {"gsm8k": 0.88}
    rerun = {"gsm8k": 0.88, "humaneval": 0.10}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []
