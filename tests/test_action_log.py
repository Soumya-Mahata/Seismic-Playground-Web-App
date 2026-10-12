"""
tests/test_action_log.py — the processing history as an immutable chronological action log
===========================================================================================
Pure tests of `modules/action_log.py`: no seismic data, no Streamlit. They pin down what
"history" means in this app:

  * the entries are in the EXACT order the actions were performed — the recommended order
    of the seismic workflow never reorders them;
  * every execution is its own entry (repeats included), and nothing is overwritten;
  * the order is the action number, not the clock;
  * displaying newest-first is a view — it never changes the log;
  * entries cannot be changed, and any tampering is detected.

The same properties through the real maths and the UI are in `test_stage_ops.py` and
`test_app_nonlinear.py`.

    pytest tests/test_action_log.py -v
"""
import csv
import dataclasses
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import action_log as al
from modules.action_log import (FAILURE, ORDERS, SUCCESS, ActionLog, format_params, format_qc,
                                one_line, report_lines, result_text)
from modules.pipeline import PIPELINE_STAGES, STAGE_INDEX

# The recommended order puts band-pass first and AGC after it; the user below does it the other way.
BRIEF = ["agc", "bandpass", "mute", "fk", "bandpass"]          # the order in the task description
LABELS = {"agc": "Amplitude Recovery (AGC)", "bandpass": "Band-Pass Filter", "mute": "Mute",
          "fk": "F-K Filter"}


def op(log, stage, params=None, inp="S1", out=None, **kw):
    return log.record("operation", stage, label=LABELS.get(stage, stage), params=params or {},
                      inputs={"data": inp} if inp else {}, output=out, **kw)


def brief_log():
    """AGC → Band-pass → Mute → F-K → Band-pass again, each reading the previous output."""
    log = ActionLog()
    prev = "S1"
    for i, stage in enumerate(BRIEF, start=2):
        op(log, stage, {"step": i}, inp=prev, out=f"S{i}")
        prev = f"S{i}"
    return log


# ─────────────────────────────────────────────────────────────────────────
# Order == execution order
# ─────────────────────────────────────────────────────────────────────────

def test_the_brief_example_stays_exactly_in_the_order_it_was_performed():
    log = brief_log()
    assert [e.action for e in log.entries()] == BRIEF
    assert [e.label for e in log.entries()] == [LABELS[s] for s in BRIEF]
    assert [e.n for e in log.entries()] == [1, 2, 3, 4, 5]
    assert [e.action for e in log.view("oldest")] == BRIEF


def test_the_recommended_order_does_not_influence_the_history():
    # sanity-check the premise: the order performed is NOT the recommended order …
    rec = [STAGE_INDEX[s] for s in BRIEF]
    assert rec != sorted(rec)
    # … and the history keeps the performed order anyway
    assert [e.action for e in brief_log().entries()] == BRIEF
    # every permutation of the stages is kept as given
    stages = [s.id for s in PIPELINE_STAGES][:12]
    log = ActionLog()
    for stage in reversed(stages):
        op(log, stage)
    assert [e.action for e in log.entries()] == list(reversed(stages))


def test_a_repeated_operation_is_a_separate_entry_and_nothing_is_overwritten():
    log = brief_log()
    first_bp, second_bp = log[2], log[5]
    assert first_bp.action == second_bp.action == "bandpass" and first_bp.n != second_bp.n
    assert len(log) == 5
    assert first_bp.output == "S3" and first_bp.input_state == "S2"      # the first one is untouched
    assert second_bp.output == "S6" and second_bp.input_state == "S5"
    # even an identical re-run (same parameters, same input) is its own entry
    op(log, "bandpass", {"step": 6}, inp="S5", out="S6", reused=True)
    assert len(log) == 6 and log[6].reused and log[6].n == 6 and not log[5].reused


def test_numbers_are_dense_and_continue_across_datasets():
    log = brief_log()
    log.new_session()
    op(log, "agc", out="S1")
    assert [e.n for e in log.entries()] == [1, 2, 3, 4, 5, 6]
    assert [e.session for e in log.entries()] == [1, 1, 1, 1, 1, 2]


class _Clock:
    """A clock that tells the same time twice, then goes backwards."""
    times = iter([datetime(2026, 1, 1, 12, 0, 5), datetime(2026, 1, 1, 12, 0, 5),
                  datetime(2026, 1, 1, 12, 0, 1), datetime(2026, 1, 1, 11, 59, 0),
                  datetime(2026, 1, 1, 12, 0, 5)])

    @classmethod
    def now(cls):
        return next(cls.times)


def test_order_comes_from_the_action_number_not_from_the_clock(monkeypatch):
    monkeypatch.setattr(al, "datetime", _Clock)
    log = brief_log()
    stamps = [e.timestamp for e in log.entries()]
    assert stamps != sorted(stamps) and len(set(stamps)) < len(stamps)   # ties AND going backwards
    assert [e.action for e in log.entries()] == BRIEF                    # order unaffected
    assert [e.action for e in log.view("oldest")] == BRIEF
    assert [e.n for e in log.view("newest")] == [5, 4, 3, 2, 1]
    assert log.verify() == []


def test_timestamps_are_recorded():
    e = brief_log()[1]
    stamp = datetime.strptime(e.timestamp, "%Y-%m-%d %H:%M:%S")
    assert abs(datetime.now() - stamp) < timedelta(minutes=1)


# ─────────────────────────────────────────────────────────────────────────
# Display order is a view
# ─────────────────────────────────────────────────────────────────────────

def test_newest_first_is_a_reversed_copy_and_the_log_itself_is_unchanged():
    log = brief_log()
    before = log.entries()
    newest, oldest = log.view("newest"), log.view("oldest")
    assert [e.n for e in newest] == [5, 4, 3, 2, 1] and [e.n for e in oldest] == [1, 2, 3, 4, 5]
    assert [e.action for e in newest] == BRIEF[::-1]
    newest.clear()                                    # mutating the returned list …
    oldest.reverse()
    assert log.entries() == before                    # … does not touch the log
    assert [e.n for e in log] == [1, 2, 3, 4, 5]
    assert log.verify() == []


def test_the_order_setting_is_validated_and_defaults_to_oldest_first():
    log = brief_log()
    assert ORDERS == ("oldest", "newest")
    assert [e.n for e in log.view()] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        log.view("alphabetical")


def test_filters_only_hide_rows_and_never_reorder_them():
    log = ActionLog()
    op(log, "agc", out="S2")
    log.record("undo", "undo", label="Undo (shot data)", inputs={"data": "S2"}, output="S1")
    op(log, "bandpass", out=None, status=FAILURE, failure_kind="blocked", error="needs shot gathers")
    op(log, "mute", out="S3", warnings=("band-pass not applied",))
    op(log, "fk", out="S4")
    assert [e.n for e in log.view(kinds=("operation",))] == [1, 3, 4, 5]
    assert [e.n for e in log.view(issues_only=True)] == [3, 4]
    assert [e.n for e in log.view("newest", issues_only=True)] == [4, 3]
    assert [e.n for e in log.view(kinds=("undo", "redo"))] == [2]
    assert [e.n for e in log.operations()] == [1, 3, 4, 5]


def test_filtering_by_dataset_session():
    log = brief_log()
    log.new_session()
    op(log, "mute", out="S1")
    assert [e.n for e in log.view(session=1)] == [1, 2, 3, 4, 5]
    assert [e.n for e in log.view(session=2)] == [6]
    assert log.sessions() == [1, 2] and log.last(session=1).n == 5 and log.last().n == 6


def test_a_new_session_is_not_started_twice_without_actions_in_between():
    log = ActionLog()
    log.new_session()
    log.new_session()
    assert log.session == 1                            # nothing recorded yet: nothing to leave behind
    op(log, "agc")
    log.new_session()
    log.new_session()
    assert log.session == 2


# ─────────────────────────────────────────────────────────────────────────
# What an entry holds
# ─────────────────────────────────────────────────────────────────────────

def test_an_entry_holds_everything_the_brief_asks_for():
    log = ActionLog()
    e = log.record("operation", "bandpass", label="Band-Pass Filter", params={"fmin": 8.0, "fmax": 60.0},
                   inputs={"data": "S2"}, output="S3", qc={"rms_in": 2.0, "rms_out": 1.0, "non_finite": 0},
                   warnings=("AGC is normally applied after band-pass.",), detail="ok")
    d = e.to_dict()
    assert d["n"] == 1 and d["timestamp"] and d["label"] == "Band-Pass Filter"      # number, time, name
    assert d["params"] == {"fmax": 60.0, "fmin": 8.0}                                # parameters
    assert d["input_state"] == "S2" and d["output_state"] == "S3"                    # input / output ids
    assert d["status"] == SUCCESS and e.ok                                           # success / failure
    assert d["qc"]["rms_out"] == 1.0                                                 # QC summary
    assert d["warnings"] == ["AGC is normally applied after band-pass."]            # warning generated


def test_a_failed_attempt_is_recorded_with_its_reason_and_no_output():
    log = ActionLog()
    e = op(log, "stack", inp="S1", out=None, status=FAILURE, failure_kind="blocked",
           error="Required input missing: NMO-corrected gathers")
    assert e.status == FAILURE and not e.ok and e.output is None
    assert e.failure_kind == "blocked" and "NMO-corrected" in e.error
    assert result_text(e) == "✗ blocked"
    assert result_text(op(log, "agc", out="S2")) == "✓ success"
    assert result_text(op(log, "agc", out="S2", reused=True)) == "♻ re-used S2"


def test_parameters_are_stored_canonically_and_not_by_reference():
    log = ActionLog()
    params = {"b": np.float64(2.5), "a": np.int64(3), "pts": [(1, 2.0), (3, 4.0)], "arr": np.arange(4.0)}
    e = log.record("operation", "fk", params=params)
    assert list(e.params) == ["a", "arr", "b", "pts"]                  # sorted keys, one spelling
    assert e.params["a"] == 3 and type(e.params["a"]) is int
    assert e.params["b"] == 2.5 and e.params["pts"] == [[1, 2.0], [3, 4.0]]
    assert "__ndarray__" in e.params["arr"]
    params["a"] = 99                                                    # the caller's dict changes later …
    params["pts"].append((9, 9))
    assert e.params["a"] == 3 and len(e.params["pts"]) == 2             # … the record does not
    e.params["a"] = -1                                                  # and a returned dict is a fresh copy
    assert e.params["a"] == 3


def test_record_rejects_unknown_kinds_and_statuses():
    log = ActionLog()
    with pytest.raises(ValueError):
        log.record("rewrite", "agc")
    with pytest.raises(ValueError):
        log.record("operation", "agc", status="maybe")
    assert len(log) == 0                                               # a refused record leaves no entry


def test_discarded_by_finds_the_later_discard_but_the_entries_remain():
    log = brief_log()                                                  # S2 … S6 produced by #1 … #5
    log.record("discard", "discard", label="Discard S4", inputs={"data": "S4"}, affected=("S4", "S5", "S6"))
    assert log.discarded_by(log[1]) is None                            # S2 survives
    assert log.discarded_by(log[3]).n == 6 and log.discarded_by(log[5]).n == 6
    assert [e.n for e in log.entries()][:5] == [1, 2, 3, 4, 5]         # the discarded states' entries stay
    other = ActionLog()
    other.record("operation", "agc", output="S4")
    other.new_session()
    other.record("discard", "discard", affected=("S4",))               # a different dataset's S4
    assert other.discarded_by(other[1]) is None


# ─────────────────────────────────────────────────────────────────────────
# Immutability and tamper evidence
# ─────────────────────────────────────────────────────────────────────────

def test_entries_are_frozen():
    e = brief_log()[2]
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.output = "S99"
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.n = 7
    with pytest.raises(dataclasses.FrozenInstanceError):
        del e.label
    assert isinstance(e.inputs, tuple) and isinstance(e.warnings, tuple) and isinstance(e.params_json, str)


def test_the_log_offers_no_way_to_change_or_remove_an_entry():
    log = brief_log()
    public = {n for n in dir(ActionLog) if not n.startswith("_")}
    writers = {n for n in public if any(w in n for w in ("remove", "delete", "pop", "clear", "insert",
                                                          "sort", "reorder", "update", "set", "edit"))}
    assert writers == set()                                            # `record` and `new_session` only
    assert isinstance(log.entries(), tuple)                            # a snapshot, not the list itself
    assert not hasattr(log, "__setitem__") and not hasattr(log, "__delitem__")
    with pytest.raises(KeyError):
        log[0]
    with pytest.raises(KeyError):
        log[6]


def test_verify_passes_on_an_untouched_log():
    assert ActionLog().verify() == []
    assert brief_log().verify() == []


def test_verify_detects_an_entry_edited_after_the_fact():
    log = brief_log()
    object.__setattr__(log[3], "params_json", '{"fmin":1}')            # sneak past `frozen`
    problems = log.verify()
    assert len(problems) == 1 and "action 3" in problems[0] and "changed" in problems[0]


def test_verify_detects_a_removed_entry():
    log = brief_log()
    del log._entries[2]
    assert log.verify()                                                # numbering gap + broken chain


def test_verify_detects_a_reordered_entry():
    log = brief_log()
    log._entries[1], log._entries[2] = log._entries[2], log._entries[1]
    problems = log.verify()
    assert any("reordered" in p for p in problems)


def test_verify_detects_an_inserted_entry():
    log = brief_log()
    forged = dataclasses.replace(log[2], n=3)
    log._entries.insert(2, forged)
    assert log.verify()


def test_verify_detects_a_rewritten_tail():
    log = brief_log()
    forged = dataclasses.replace(log[5], action="fk", label="F-K Filter")   # swap the last step's identity
    log._entries[4] = forged
    assert any("action 5" in p for p in log.verify())


# ─────────────────────────────────────────────────────────────────────────
# Text and file forms keep the order
# ─────────────────────────────────────────────────────────────────────────

def test_the_text_report_lists_actions_in_the_order_performed():
    log = brief_log()
    text = "\n".join(report_lines(log.entries()))
    positions = [text.index(f"#{i} ") for i in range(1, 6)]
    assert positions == sorted(positions)
    assert text.count("Band-Pass Filter") == 2                         # both band-passes listed
    assert "input: data=S2   →   output: S3" in text


def test_one_line_summaries_carry_number_label_and_states():
    log = ActionLog()
    e = op(log, "agc", inp="S2", out="S3")
    assert one_line(e).startswith("#1 ✓ Amplitude Recovery (AGC)") and "S2 → S3" in one_line(e)
    f = op(log, "stack", inp="S2", out=None, status=FAILURE, failure_kind="blocked", error="x")
    assert one_line(f).startswith("#2 ✗")


def test_format_helpers():
    assert format_params({"window": 500, "mode": "rms", "g": 0.123456, "pts": [1, 2, 3]}) == \
        "window=500, mode=rms, g=0.1235, pts=[3]"
    assert format_qc({"rms_in": 2.0, "rms_out": 1.0, "non_finite": 0}) == "RMS 2 → 1 (×0.5) · all finite"
    assert "⚠ 3 non-finite sample(s)" in format_qc({"non_finite": 3})
    assert format_qc({}) == ""


def test_the_csv_export_is_chronological_with_the_true_action_number():
    from utils import io_export
    log = brief_log()
    rows = list(csv.DictReader(io.StringIO(io_export.action_log_csv_bytes(log).decode())))
    assert [r["action_number"] for r in rows] == ["1", "2", "3", "4", "5"]
    assert [r["operation"] for r in rows] == [LABELS[s] for s in BRIEF]
    assert rows[0]["input_state"] == "S1" and rows[0]["output_state"] == "S2"
    rev = list(csv.DictReader(io.StringIO(io_export.action_log_csv_bytes(log, order="newest").decode())))
    assert [r["action_number"] for r in rev] == ["5", "4", "3", "2", "1"]   # a view; numbers still say the truth


def test_the_json_export_carries_the_chain_and_reports_integrity():
    from utils import io_export
    log = brief_log()
    payload = json.loads(io_export.action_log_json_bytes(log))
    assert [a["action"] for a in payload["actions"]] == BRIEF and payload["integrity_problems"] == []
    assert payload["actions"][1]["prev_hash"] == payload["actions"][0]["hash"]
