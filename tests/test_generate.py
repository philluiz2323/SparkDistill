from teacher.generate import _iter_prompts


def _write(tmp_path, lines):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_iter_prompts_limit_counts_only_yielded_records(tmp_path):
    # A leading blank line must not consume the --limit budget: the flag is
    # documented as "only sample the first N prompts", so N non-blank records
    # should be yielded regardless of interspersed blank lines.
    path = _write(tmp_path, ["", '{"prompt": "a"}', '{"prompt": "b"}'])

    records = list(_iter_prompts(path, limit=2))

    assert [r["prompt"] for r in records] == ["a", "b"]


def test_iter_prompts_limit_ignores_interspersed_blank_lines(tmp_path):
    path = _write(
        tmp_path,
        ['{"prompt": "a"}', "", "   ", '{"prompt": "b"}', '{"prompt": "c"}'],
    )

    records = list(_iter_prompts(path, limit=2))

    assert [r["prompt"] for r in records] == ["a", "b"]


def test_iter_prompts_without_limit_yields_all_non_blank(tmp_path):
    path = _write(tmp_path, ['{"prompt": "a"}', "", '{"prompt": "b"}'])

    records = list(_iter_prompts(path, limit=None))

    assert [r["prompt"] for r in records] == ["a", "b"]
