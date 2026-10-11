"""
tests/test_state_graph.py — the data-state graph (pure Python, no data, no UI)
================================================================================
    pytest tests/test_state_graph.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.state_graph import (GraphError, ProcessingGraph, array_digest, canonical, jsonable,
                                 make_fingerprint)


def chain():
    """S0 raw → S1 shots → S2 AGC → S3 band-pass → S4 mute (the user's example, one kind: shot)."""
    g = ProcessingGraph()
    s0 = g.add(kind="raw", op="upload", params={"source": "x"})
    s1 = g.add(kind="shot", op="shot_gathers", inputs={"data": s0.id}, root="self")
    s2 = g.add(kind="shot", op="agc", params={"window": 500}, inputs={"data": s1.id}, root=s1.id)
    s3 = g.add(kind="shot", op="bandpass", params={"fmin": 8.0, "fmax": 60.0},
               inputs={"data": s2.id}, root=s1.id)
    s4 = g.add(kind="shot", op="mute", params={"t_top": 0.1}, inputs={"data": s3.id}, root=s1.id)
    return g, [s0, s1, s2, s3, s4]


# ─────────────────────────────────────────────────────────────────────────
# identity
# ─────────────────────────────────────────────────────────────────────────

def test_ids_are_sequential_from_s0_and_never_reused():
    g, st = chain()
    assert [s.id for s in st] == ["S0", "S1", "S2", "S3", "S4"]
    g.discard("S4")
    assert g.peek_id() == "S5"
    assert g.add(kind="shot", op="mute", inputs={"data": "S3"}).id == "S5"
    assert "S4" not in g


def test_each_state_records_what_it_consumed_and_its_operation_and_parameters():
    g, st = chain()
    s3 = g["S3"]
    assert (s3.op, s3.params, s3.inputs) == ("bandpass", {"fmax": 60.0, "fmin": 8.0}, {"data": "S2"})
    assert g["S2"].params == {"window": 500}


def test_version_counts_states_made_by_the_same_operation():
    g, st = chain()
    again = g.add(kind="shot", op="agc", params={"window": 250}, inputs={"data": "S4"})
    assert (st[2].version, again.version) == (1, 2)
    assert g.add(kind="shot", op="bandpass", inputs={"data": again.id}).version == 2


def test_fingerprint_is_deterministic_and_ignores_key_order_and_numpy_types():
    a = make_fingerprint("agc", "", {"window": 500, "mode": "rms"}, {"data": "abc"})
    b = make_fingerprint("agc", "", {"mode": "rms", "window": np.int64(500)}, {"data": "abc"})
    assert a == b and len(a) == 12
    assert make_fingerprint("agc", "", {"window": 500.0}, {"data": "abc"}) != \
        make_fingerprint("agc", "", {"window": 501.0}, {"data": "abc"})


def test_fingerprint_depends_on_the_whole_lineage_not_just_the_last_step():
    g1, _ = chain()
    g2 = ProcessingGraph()
    s0 = g2.add(kind="raw", op="upload", params={"source": "x"})
    s1 = g2.add(kind="shot", op="shot_gathers", inputs={"data": s0.id}, root="self")
    s3 = g2.add(kind="shot", op="bandpass", params={"fmin": 8.0, "fmax": 60.0},
                inputs={"data": s1.id}, root=s1.id)           # same band-pass, without the AGC before it
    assert g1["S3"].fingerprint != s3.fingerprint
    # ... and the same lineage in a second graph reproduces it exactly
    g3, _ = chain()
    assert [s.fingerprint for s in g1] == [s.fingerprint for s in g3]


def test_fingerprint_ignores_diagnostics_and_notes():
    g = ProcessingGraph()
    s0 = g.add(kind="raw", op="upload")
    a = g.add(kind="shot", op="despike", params={"w": 15}, inputs={"data": s0.id},
              info={"flagged": 3}, notes="x")
    b = g.add(kind="shot", op="despike", params={"w": 15}, inputs={"data": s0.id},
              info={"flagged": 99}, notes="y")
    assert a.fingerprint == b.fingerprint and a.id != b.id
    assert g.find(a.fingerprint).id == b.id            # most recently touched wins


def test_canonical_handles_sets_tuples_arrays_nan_and_rejects_unreproducible_objects():
    assert canonical({"b": (1, 2), "a": {3, 1}}) == {"a": [1, 3], "b": [1, 2]}
    assert canonical(np.float32(2.5)) == 2.5 and canonical(float("nan")) == "nan"
    arr = canonical(np.arange(6.0).reshape(2, 3))
    assert arr["shape"] == [2, 3] and arr["__ndarray__"] == array_digest(np.arange(6.0).reshape(2, 3))
    assert array_digest(np.zeros(4)) != array_digest(np.ones(4))
    with pytest.raises(TypeError):
        canonical({"f": lambda x: x})
    assert isinstance(jsonable({"f": object()})["f"], str)       # never raises for diagnostics


def test_unknown_kind_or_missing_input_is_rejected():
    g = ProcessingGraph()
    with pytest.raises(ValueError):
        g.add(kind="nonsense", op="x")
    with pytest.raises(GraphError, match="does not exist"):
        g.add(kind="shot", op="x", inputs={"data": "S9"})
    with pytest.raises(GraphError, match="No data state"):
        g["S9"]


# ─────────────────────────────────────────────────────────────────────────
# current / select
# ─────────────────────────────────────────────────────────────────────────

def test_current_is_the_latest_state_of_the_kind_and_select_moves_it():
    g, st = chain()
    assert g.current("shot").id == "S4" and g.current("raw").id == "S0" and g.current("cdp") is None
    g.select("S2")
    assert g.current("shot").id == "S2"
    assert g.latest("mute").id == "S4"                  # the op pointer is unaffected by data selection of AGC
    g.select("S4")
    assert g.current("shot").id == "S4"


def test_running_an_operation_on_an_older_state_branches_and_becomes_current():
    g, st = chain()
    branch = g.add(kind="shot", op="decon", inputs={"data": "S2"}, root="S1")
    assert g.current("shot").id == branch.id
    assert [d.id for d in g.dependents("S2")] == ["S3", "S4", branch.id]
    assert g["S4"] in g.states()                         # nothing was overwritten


def test_repeated_operation_makes_a_second_state_not_a_replacement():
    g, st = chain()
    second = g.add(kind="shot", op="agc", params={"window": 250}, inputs={"data": "S4"})
    assert {s.id for s in g.of_op("agc")} == {"S2", second.id}
    assert g.latest("agc").id == second.id
    assert [s.op for s in g.lineage(second.id)] == ["upload", "shot_gathers", "agc", "bandpass", "mute", "agc"]


# ─────────────────────────────────────────────────────────────────────────
# undo / redo
# ─────────────────────────────────────────────────────────────────────────

def test_undo_walks_back_through_the_lineage_and_redo_forward_without_deleting_anything():
    g, st = chain()
    assert g.undo("shot").id == "S3" and g.current("shot").id == "S3"
    assert g.undo("shot").id == "S2"
    assert len(g) == 5                                   # nothing was removed
    assert g.redo_target("shot").id == "S3"
    assert g.redo("shot").id == "S3" and g.redo("shot").id == "S4"
    assert g.redo("shot") is None and g.current("shot").id == "S4"
    assert g.redo_target("shot") is None


def test_undo_stops_at_the_start_of_the_kind():
    g, st = chain()
    for _ in range(3):
        g.undo("shot")                                   # S4 → S3 → S2 → S1
    assert g.current("shot").id == "S1"
    assert not g.can_undo("shot") and g.undo("shot") is None   # S1's input is raw data, another kind
    assert g.undo("raw") is None


def test_a_new_operation_after_undo_branches_and_ends_the_redo_history():
    g, st = chain()
    g.undo("shot")
    assert g.can_redo("shot")
    new = g.add(kind="shot", op="balance", inputs={"data": "S3"}, root="S1")
    assert not g.can_redo("shot") and g.current("shot").id == new.id
    assert g["S4"].id == "S4"                            # the undone branch is still recorded


def test_select_ends_the_redo_history_too():
    g, st = chain()
    g.undo("shot")
    g.select("S2")
    assert not g.can_redo("shot")


def test_redo_skips_states_discarded_meanwhile():
    g, st = chain()
    g.undo("shot"); g.undo("shot")                       # redo stack: S4, S3
    g.discard("S3")                                      # removes S3 and S4
    assert g.redo("shot") is None


# ─────────────────────────────────────────────────────────────────────────
# lineage, dependents, discard
# ─────────────────────────────────────────────────────────────────────────

def multi_input():
    g, st = chain()
    cdp = g.add(kind="cdp", op="cdp_sort", inputs={"data": "S4"}, root="self")        # S5
    vel = g.add(kind="velocity", op="vel1", inputs={"data": cdp.id}, root=cdp.id)     # S6
    nmo = g.add(kind="nmo", op="nmo1", inputs={"data": cdp.id, "velocity": vel.id}, root=cdp.id)  # S7
    return g, cdp, vel, nmo


def test_dependents_follow_every_input_role():
    g, cdp, vel, nmo = multi_input()
    assert [d.id for d in g.dependents(vel.id)] == [nmo.id]          # via the 'velocity' role
    assert [d.id for d in g.dependents("S2")] == ["S3", "S4", "S5", "S6", "S7"]
    assert g.dependents(nmo.id) == []


def test_ancestors_and_lineage_differ_by_roles_followed():
    g, cdp, vel, nmo = multi_input()
    assert [a.id for a in g.ancestors(nmo.id)] == ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    assert [s.id for s in g.lineage(nmo.id)] == ["S0", "S1", "S2", "S3", "S4", "S5", "S7"]   # 'data' chain only
    assert g.is_ancestor("S6", nmo.id) and not g.is_ancestor(nmo.id, "S6")


def test_discard_removes_the_state_and_only_what_was_built_on_it():
    g, cdp, vel, nmo = multi_input()
    removed = g.discard(vel.id)
    assert set(removed) == {vel.id, nmo.id} and vel.id not in g and cdp.id in g and "S3" in g
    assert removed[-1] == vel.id                         # dependents come first
    assert g.discard("S3") and len(g) == 3               # S0, S1, S2 remain
    assert g.events[-1]["event"] == "discard"


def test_reset_forgets_everything_and_restarts_at_s0():
    g, *_ = multi_input()
    g.reset()
    assert len(g) == 0 and g.peek_id() == "S0" and g.events == []


# ─────────────────────────────────────────────────────────────────────────
# reproducibility
# ─────────────────────────────────────────────────────────────────────────

def test_recipe_lists_exactly_the_ancestors_in_replayable_order():
    g, cdp, vel, nmo = multi_input()
    r = g.recipe(nmo.id)
    assert [s["id"] for s in r] == ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    assert r[-1] == {"id": "S7", "kind": "nmo", "op": "nmo1", "variant": "", "params": {},
                     "inputs": {"data": "S5", "velocity": "S6"}, "fingerprint": nmo.fingerprint}
    other = g.add(kind="shot", op="decon", inputs={"data": "S2"})
    assert [s["id"] for s in g.recipe(other.id)] == ["S0", "S1", "S2", other.id]   # S3/S4 are not in its past
    json.dumps(r)                                        # a recipe is plain JSON


def test_export_is_json_able_and_reports_the_current_pointers():
    g, cdp, vel, nmo = multi_input()
    g.add(kind="section", op="stack", inputs={"data": nmo.id}, info={"fold": np.arange(3)})
    out = g.export()
    json.dumps(out)
    assert out["current"]["cdp"] == cdp.id and out["current"]["shot"] == "S4"
    assert out["states"][3]["op"] == "bandpass" and out["states"][3]["inputs"] == {"data": "S2"}
