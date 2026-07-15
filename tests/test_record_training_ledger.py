import json

import eval.record_training_ledger as cli


def test_main_writes_ledger_entry(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    report = {
        "verified": True,
        "label": "eval:BASELINE",
        "best_benchmark": None,
        "best_pct_delta": None,
        "regressions": [],
        "run_id": "run-1",
    }
    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (report, None, None))

    pr_body_file = tmp_path / "pr_body.md"
    pr_body_file.write_text("Proof-bundle URL: https://huggingface.co/org/proof-repo")
    ledger_path = tmp_path / "ledger.jsonl"

    rc = cli.main(
        [
            "--pr-url",
            "https://github.com/org/repo/pull/1",
            "--pr-body-file",
            str(pr_body_file),
            "--ledger-path",
            str(ledger_path),
        ]
    )
    assert rc == 0
    entry = json.loads(ledger_path.read_text().splitlines()[0])
    assert entry["run_id"] == "run-1"


def test_main_returns_nonzero_on_issue(tmp_path, monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "_download_and_verify_bundle", lambda *a, **k: (None, None, "download failed"))

    pr_body_file = tmp_path / "pr_body.md"
    pr_body_file.write_text("Proof-bundle URL: https://huggingface.co/org/proof-repo")

    rc = cli.main(
        [
            "--pr-url",
            "https://github.com/org/repo/pull/1",
            "--pr-body-file",
            str(pr_body_file),
            "--ledger-path",
            str(tmp_path / "ledger.jsonl"),
        ]
    )
    assert rc == 1
