from eval.score import score


def test_score_improvement_gets_expected_tier():
    candidate = {"gsm8k": 88.0, "humaneval": 80.0}
    frontier = {"gsm8k": 80.0, "humaneval": 80.0}
    report = score(candidate, frontier)
    assert report["label"] == "eval:L"  # (88-80)/80 = 10.0% -> L band
    assert report["best_benchmark"] == "gsm8k"
    assert report["regressions"] == []


def test_score_rejects_on_regression_beyond_floor():
    candidate = {"gsm8k": 88.0, "humaneval": 70.0}
    frontier = {"gsm8k": 80.0, "humaneval": 80.0}
    report = score(candidate, frontier)
    assert report["label"] == "eval:REJECT"
    assert "regression-humaneval" in report["regressions"]


def test_score_none_below_minimum_tier():
    candidate = {"gsm8k": 80.5}
    frontier = {"gsm8k": 80.0}
    report = score(candidate, frontier)
    assert report["label"] == "eval:none"
