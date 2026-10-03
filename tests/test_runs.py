"""Run helpers should preserve prior results and resolve numeric run order."""

from v2.utils.runs import find_latest_result, make_run_dir


def test_new_run_preserves_previous_outputs_and_increments_numerically(tmp_path):
    previous = tmp_path / "runs" / "run9" / "results"
    previous.mkdir(parents=True)
    saved = previous / "scores.csv"
    saved.write_text("accuracy\n0.8\n", encoding="utf-8")
    (tmp_path / "runs" / "run_notes").mkdir()
    (tmp_path / "runs" / "run99").write_text("not a directory", encoding="utf-8")

    created = make_run_dir(tmp_path, notebook_label="toy experiment")
    assert created.name == "run10"
    assert (created / "figures").is_dir()
    assert (created / "results").is_dir()
    assert "toy experiment" in (created / "README.md").read_text(encoding="utf-8")
    assert saved.read_text(encoding="utf-8") == "accuracy\n0.8\n"
    assert make_run_dir(tmp_path).name == "run11"


def test_result_lookup_skips_incomplete_runs_and_uses_numeric_order(tmp_path):
    filename = "scores.csv"
    for number in [2, 3, 10]:
        results = tmp_path / "runs" / f"run{number}" / "results"
        results.mkdir(parents=True)
        if number != 10:
            (results / filename).write_text(str(number), encoding="utf-8")
    assert find_latest_result(tmp_path, filename) == tmp_path / "runs/run3/results" / filename
    newest = tmp_path / "runs/run10/results" / filename
    newest.write_text("10", encoding="utf-8")
    assert find_latest_result(tmp_path, filename) == newest
    assert find_latest_result(tmp_path, "missing.csv") is None


def test_result_lookup_supports_legacy_layout(tmp_path):
    legacy = tmp_path / "scores.csv"
    legacy.write_text("legacy", encoding="utf-8")
    assert find_latest_result(tmp_path, "scores.csv") == legacy
    make_run_dir(tmp_path)
    assert find_latest_result(tmp_path, "scores.csv") == legacy
