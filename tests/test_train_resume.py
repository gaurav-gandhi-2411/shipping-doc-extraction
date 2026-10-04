"""Kill / resume equivalence of the fine-tune loop, and the ``metrics.jsonl`` step invariant.

Background (reports/finetune_plan.md "Smoke measurement"): the first L4 smoke logged 29 lines for a
20-step target because session 1 died at step 9, before the first checkpoint, and session 2's
restart appended to its file. Everything here runs on the tiny CPU model of ``test_train``; CUDA
RNG restoration (``torch.cuda.set_rng_state_all``) is therefore NOT exercised.
"""

from __future__ import annotations

import ast
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_train import (
    IMG_ID,
    FakeProcessor,
    Kill,
    ListSource,
    assert_same,
    build_model,
    hidden_fn,
    kill_at,
    make_cfg,
    make_trainer,
    snapshot,
    write_corpus,
)

from shipdoc import metricslog as ml
from shipdoc import train as tr
from shipdoc import trainset as ts

torch = pytest.importorskip("torch")

KEYS = ("step", "epoch", "loss", "grad_norm", "lr", "n_micro", "eval_loss", "eval_loss_epoch")
#: micro-batches per optimizer step of ListSource(7), accum 3, 2 epochs: [3, 3, 1] twice
MICRO = [3, 3, 1, 3, 3, 1]


class RecSource(ListSource):
    """`ListSource` that logs every ``get`` (sample, epoch, augmentation seeds, RNG draws) and can
    be told to die inside the k-th call (1-based), i.e. in the middle of gradient accumulation.

    The python / numpy draws make the log sensitive to ANY restored-RNG mistake.
    """

    def __init__(
        self, log: list[tuple[Any, ...]], kill_call: int | None = None, n: int = 7
    ) -> None:
        super().__init__(n)
        self.log, self.kill_call, self.calls = log, kill_call, 0

    def get(self, index: int, epoch: int) -> dict[str, Any]:
        self.calls += 1
        if self.kill_call == self.calls:
            raise Kill
        doc = f"doc{index}"
        self.log.append((
            doc, index % 2, epoch,
            ts.sample_seed(42, doc, index % 2, epoch, "occ"),
            ts.sample_seed(42, doc, index % 2, epoch, "scan"),
            random.random(), float(np.random.rand()),
        ))  # fmt: skip
        return super().get(index, epoch)


def life(run: Path, source: ListSource, **kw: Any) -> tr.Trainer:
    """A fresh process: new model, seeds as ``train.main`` sets them (python / numpy / torch)."""
    t = make_trainer(run, source=source, **kw)
    tr.seed_all(42)
    return t


def run_reference(tmp: Path) -> tuple[tr.Trainer, list[tuple[Any, ...]]]:
    log: list[tuple[Any, ...]] = []
    ref = life(tmp / "ref", RecSource(log))
    ref.fit()
    return ref, log


def file_steps(run: Path) -> list[int]:
    lines = (run / "metrics.jsonl").read_text().splitlines()
    return [json.loads(x)["step"] for x in lines]


def assert_equivalent(
    ref: tr.Trainer, ref_log: list[Any], run: Path, log1: list[Any], log2: list[Any],
    resumed: tr.Trainer, ckpt_step: int,
) -> None:  # fmt: skip
    """(a) step sequence (b) sample order incl. aug seeds / RNG draws (c) final adapter
    (d) metrics.jsonl has exactly total lines, no duplicate step."""
    pick = lambda h: [{k: r.get(k) for k in KEYS} for r in h]  # noqa: E731
    assert [r["step"] for r in resumed.history] == [r["step"] for r in ref.history]  # (a)
    assert pick(resumed.history) == pick(ref.history)
    done = sum(MICRO[:ckpt_step])
    assert log1 == ref_log[: len(log1)]  # life 1 saw the reference order up to its death
    assert log2 == ref_log[done:]  # (b) life 2 replays from the checkpoint, identically
    assert_same(snapshot(ref), snapshot(resumed))  # (c)
    assert file_steps(run) == list(range(1, ref.total + 1))  # (d)
    assert len(resumed.history) == resumed.total == len(ref.history)


def ckpt_step_before(step_done: int, every: int = 3) -> int:
    return step_done - step_done % every


@pytest.mark.parametrize("kill_step", [1, 2, 3, 4, 5])
def test_kill_after_a_step_resumes_identically(tmp_path: Path, kill_step: int) -> None:
    """Steps 1-2: no checkpoint yet (the smoke anomaly); 3: right after one; 5: right before 6."""
    ref, ref_log = run_reference(tmp_path)
    run = tmp_path / "killed"
    log1: list[tuple[Any, ...]] = []
    first = life(run, RecSource(log1), on_step=kill_at(kill_step))
    with pytest.raises(Kill):
        first.fit()
    log2: list[tuple[Any, ...]] = []
    again = life(run, RecSource(log2), model=build_model())
    again.fit()
    assert_equivalent(ref, ref_log, run, log1, log2, again, ckpt_step_before(kill_step))


@pytest.mark.parametrize("kill_call", [2, 5, 7, 8, 9, 13])
def test_kill_inside_gradient_accumulation_resumes_identically(
    tmp_path: Path, kill_call: int
) -> None:
    """Calls 2/5/9: mid-accumulation; 7: the partial batch of step 3, just before its checkpoint;
    8: first micro-batch right after the step-3 checkpoint; 13: last micro-batch of step 5."""
    ref, ref_log = run_reference(tmp_path)
    run = tmp_path / "killed"
    log1: list[tuple[Any, ...]] = []
    first = life(run, RecSource(log1, kill_call=kill_call))
    with pytest.raises(Kill):
        first.fit()
    step_done = sum(1 for i in range(len(MICRO)) if sum(MICRO[: i + 1]) < kill_call)
    assert first.step == step_done  # the interrupted step left no trace in the optimizer state
    log2: list[tuple[Any, ...]] = []
    again = life(run, RecSource(log2), model=build_model())
    again.fit()
    assert_equivalent(ref, ref_log, run, log1, log2, again, ckpt_step_before(step_done))


def test_two_kills_in_a_row_and_a_third_life(tmp_path: Path) -> None:
    ref, ref_log = run_reference(tmp_path)
    run = tmp_path / "killed"
    logs: list[list[tuple[Any, ...]]] = [[], [], []]
    a = life(run, RecSource(logs[0], kill_call=10))  # dies in step 4, ckpt 3
    with pytest.raises(Kill):
        a.fit()
    b = life(run, RecSource(logs[1]), model=build_model(), on_step=kill_at(4))
    with pytest.raises(Kill):
        b.fit()  # replays step 4 from the step-3 checkpoint and dies again after it
    c = life(run, RecSource(logs[2]), model=build_model())
    c.fit()
    assert logs[1] == ref_log[sum(MICRO[:3]) : sum(MICRO[:4])]
    assert logs[2] == ref_log[sum(MICRO[:3]) :]
    assert_same(snapshot(ref), snapshot(c))
    assert file_steps(run) == list(range(1, 7))


def test_resume_restores_every_piece_of_state(tmp_path: Path) -> None:
    run = tmp_path / "run"
    t = make_trainer(run, precision="fp16", on_step=kill_at(4))
    with pytest.raises(Kill):
        t.fit()
    saved = torch.load(run / "ckpt" / "step_000003" / "trainer_state.pt", weights_only=False)
    fresh = make_trainer(run, model=build_model(), precision="fp16")
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    assert fresh.resume() is True
    assert fresh.step == saved["step"] == 3 and len(fresh.history) == 3
    assert fresh.sched.state_dict() == saved["scheduler"]
    assert fresh.sched.state_dict()["last_epoch"] == 3
    assert fresh.opt.param_groups[0]["lr"] == saved["optimizer"]["param_groups"][0]["lr"]
    assert float(fresh.scaler.get_scale()) == saved["scaler"]["scale"]
    assert fresh.opt.state_dict()["state"].keys() == saved["optimizer"]["state"].keys()
    assert random.getstate() == saved["rng"]["python"]
    assert np.array_equal(np.random.get_state()[1], saved["rng"]["numpy"][1])
    assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
    # the sampler position is a pure function of the restored step (epoch 1, batch 0)
    assert tr.step_indices(7, 3, 42, fresh.step)[0] == saved["epoch"] == 1
    assert saved["sampler_pos"] == 0


def test_resume_rejects_an_inconsistent_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    make_trainer(run).fit()
    path = run / "ckpt" / "step_000006" / "trainer_state.pt"
    state = torch.load(path, weights_only=False)
    state["sampler_pos"] += 3  # the data-loader position no longer matches the step
    torch.save(state, path)
    with pytest.raises(tr.TrainError, match="inconsistent"):
        make_trainer(run).resume()
    state["sampler_pos"] -= 3
    state["history"] = state["history"][:-1]  # history shorter than the step counter
    torch.save(state, path)
    with pytest.raises(tr.TrainError, match="inconsistent"):
        make_trainer(run).resume()


def test_a_trainer_that_is_ahead_of_every_checkpoint_refuses_to_restart_in_process(
    tmp_path: Path,
) -> None:
    t = make_trainer(tmp_path, cfg=make_cfg(ckpt_every=0), on_step=kill_at(2))
    with pytest.raises(Kill):
        t.fit()
    with pytest.raises(tr.TrainError, match="build a new Trainer"):
        t.fit()  # weights are at step 2 but nothing could restore step 0


def test_end_to_end_page_dataset_sample_order_incl_augmentation_seeds(tmp_path: Path) -> None:
    """The real data path: (doc, page, epoch, occlusion seed, scan seed) per micro-batch."""
    prepared = write_corpus(tmp_path / "data")
    specs = prepared.page_specs(prepared.plans)
    cfg = make_cfg(occlusion_rate=0.5, scan_prob=0.0, grad_accum=2, ckpt_every=4, eval_every=4,
                   epochs=2, loss_chunk=64)  # fmt: skip

    class Logged(tr.PageDataset):
        def __init__(self, log: list[Any], kill_call: int | None = None) -> None:
            super().__init__(prepared, specs, cfg, FakeProcessor(), IMG_ID)
            self.log, self.kill_call, self.calls = log, kill_call, 0

        def get(self, index: int, epoch: int) -> dict[str, Any]:
            self.calls += 1
            if self.kill_call == self.calls:
                raise Kill
            sp = self.specs[index]
            seeds = [
                ts.sample_seed(42, sp.doc_id, sp.page_index, epoch, t) for t in ("occ", "scan")
            ]
            self.log.append((sp.doc_id, sp.page_index, epoch, *seeds))
            return super().get(index, epoch)

    def trainer(root: Path, log: list[Any], kill_call: int | None = None) -> tr.Trainer:
        t = tr.Trainer(build_model(), Logged(log, kill_call), cfg, hidden_fn, root,
                       signature="f", grad_accum=2)  # fmt: skip
        tr.seed_all(42)
        return t

    ref_log: list[Any] = []
    ref = trainer(tmp_path / "ref", ref_log)
    ref.fit()
    log1: list[Any] = []
    with pytest.raises(Kill):
        trainer(tmp_path / "k", log1, kill_call=10).fit()  # 1st micro-batch of step 6; ckpt at 4
    log2: list[Any] = []
    again = trainer(tmp_path / "k", log2)
    again.fit()
    assert log1 == ref_log[: len(log1)]
    assert log2 == ref_log[8:]  # micro-batches per step 2,2,2,2,1,...: 8 done at the step-4 ckpt
    assert_same(snapshot(ref), snapshot(again))
    assert file_steps(tmp_path / "k") == list(range(1, ref.total + 1))


# --------------------------------------------------------------------------------------------
# The smoke stage honours its own target
# --------------------------------------------------------------------------------------------


def smoke_like(run: Path, **kw: Any) -> tr.Trainer:
    """The real smoke's shape: 80 pages, accum 2 (40 steps per epoch) but a 20-step target."""
    smoke = tr.SmokeCfg(steps=20, grad_accum=2, first_last_k=5, min_rel_decrease=0.05)
    cfg = make_cfg(smoke=smoke, ckpt_every=10, eval_every=0, epochs=2)
    t = make_trainer(run, cfg=cfg, total_steps=20, grad_accum=2, source=ListSource(80),
                     eval_source=None, vram_fn=lambda: 1, **kw)  # fmt: skip
    tr.seed_all(42)
    return t


def run_smoke_ok(t: tr.Trainer) -> list[tr.Check]:
    return tr.run_smoke(t, expected_modules=6, expected_trainable=tr.count_trainable(t.model),
                        vram_budget=2**20, require_vram=True, longest_index=0)  # fmt: skip


@pytest.mark.parametrize("kill_step", [9, 10, 15, 19])
def test_smoke_stops_at_20_and_has_20_lines_after_any_resume(
    tmp_path: Path, kill_step: int
) -> None:
    """9 reproduces the real incident: killed before the first checkpoint (step 10)."""
    ref = smoke_like(tmp_path / "ref")
    run_smoke_ok(ref)
    run = tmp_path / "run"
    with pytest.raises(Kill):
        run_smoke_ok(smoke_like(run, on_step=kill_at(kill_step)))
    assert len(file_steps(run)) == kill_step  # life 1 left exactly the steps it finished
    again = smoke_like(run, model=build_model())  # a new process
    checks = run_smoke_ok(again)
    assert again.step == again.total == 20 and len(again.history) == 20  # (e)
    assert file_steps(run) == list(range(1, 21))  # (d) 20 lines, not 20 + kill_step
    assert next(c for c in checks if c.name == "metrics_file").status == "pass"
    assert_same(snapshot(ref), snapshot(again))
    assert [r["loss"] for r in again.history] == [r["loss"] for r in ref.history]
    assert max(r["step"] for r in again.history) == 20  # the target is never recomputed upwards


# --------------------------------------------------------------------------------------------
# metrics.jsonl: the real anomaly, reader and repair
# --------------------------------------------------------------------------------------------


def anomaly_rows() -> list[dict[str, Any]]:
    """Shaped like the real file (29 lines for a 20-step target: steps 1..9 then 1..20),
    SYNTHETIC numbers."""

    def row(step: int, session: int) -> dict[str, Any]:
        return {"step": step, "epoch": 0, "loss": round(1.0 / (step + session), 6),
                "grad_norm": 0.5, "lr": 1e-4, "n_micro": 2, "scaler_skipped": False,
                "seconds": 17.0 + session}  # fmt: skip

    return [row(s, 1) for s in range(1, 10)] + [row(s, 2) for s in range(1, 21)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_the_real_anomaly_is_detected_by_the_strict_reader(tmp_path: Path) -> None:
    rows = anomaly_rows()
    path = tmp_path / "metrics.jsonl"
    write_rows(path, rows)
    assert len(rows) == 29  # what the old banner printed as "29 / 20"
    with pytest.raises(ml.MetricsAnomaly, match="more than once"):
        ml.read_metrics(path, total=20)
    check = tr.metrics_file_check(path, rows[9:], 20)
    assert check.status == "fail" and "29 lines for a 20-step run" in check.detail
    fixed = ml.read_metrics(path, total=20, repair=True)
    assert fixed.repaired and fixed.n_lines == 29
    assert [r["step"] for r in fixed.rows] == list(range(1, 21))
    assert fixed.rows == rows[9:]  # last writer wins: the second session's lines


def test_the_reader_accepts_clean_files_and_repairs_a_replay_after_a_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    assert ml.read_metrics(path).rows == []  # absent file = empty
    write_rows(path, [{"step": s} for s in range(1, 6)])
    clean = ml.read_metrics(path, total=5)
    assert not clean.repaired and clean.n_lines == 5
    # killed after step 12 with a checkpoint at 10: 11 and 12 are written again by the restart
    write_rows(path, [{"step": s} for s in [*range(1, 13), *range(11, 16)]])
    assert [r["step"] for r in ml.read_metrics(path, total=15, repair=True).rows] == list(
        range(1, 16)
    )
    write_rows(path, [{"step": 1}, {"step": 2}, {"step": 4}])  # a gap cannot be repaired
    with pytest.raises(ml.MetricsAnomaly, match="not repairable"):
        ml.read_metrics(path, repair=True)
    write_rows(path, [{"step": s} for s in range(1, 8)])
    with pytest.raises(ml.MetricsAnomaly, match="7 lines for a 5-step run"):
        ml.read_metrics(path, total=5)
    path.write_text('{"step": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ml.MetricsAnomaly, match="not JSON"):
        ml.read_metrics(path)
    path.write_text('{"loss": 1}\n', encoding="utf-8")
    with pytest.raises(ml.MetricsAnomaly, match="'step'"):
        ml.read_metrics(path)


def test_the_writer_is_idempotent_and_refuses_gaps(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    for s in (1, 2, 3):
        ml.append_metric(path, {"step": s, "loss": float(s)})
    ml.append_metric(path, {"step": 2, "loss": 20.0})  # the same step again replaces it
    assert [(r["step"], r["loss"]) for r in ml.read_metrics(path).rows] == [(1, 1.0), (2, 20.0)]
    ml.append_metric(path, {"step": 2, "loss": 20.0})  # and again: no change
    assert [r["step"] for r in ml.read_metrics(path).rows] == [1, 2]
    with pytest.raises(ml.MetricsAnomaly, match="gap"):
        ml.append_metric(path, {"step": 5})
    with pytest.raises(ml.MetricsAnomaly, match="gap"):
        ml.append_metric(tmp_path / "new.jsonl", {"step": 2})


def test_a_restart_without_checkpoint_discards_the_dead_sessions_lines(tmp_path: Path) -> None:
    """The fix itself: stale lines from a session that died before step 10 never survive."""
    stale = anomaly_rows()[:9]  # session 1: steps 1..9
    run = tmp_path / "run"
    run.mkdir()
    write_rows(run / "metrics.jsonl", stale)
    t = make_trainer(run, cfg=make_cfg(ckpt_every=3), source=ListSource(10), total_steps=6,
                     grad_accum=2)  # fmt: skip
    t.fit()
    assert file_steps(run) == [1, 2, 3, 4, 5, 6]
    discarded = [json.loads(x) for x in (run / "metrics_discarded.jsonl").read_text().splitlines()]
    assert [r["step"] for r in discarded] == list(range(1, 10))  # kept for diagnosis
    assert {r["discarded_at_step"] for r in discarded} == {0}
    # a second, clean restart of the finished run changes nothing and discards nothing
    again = make_trainer(run, cfg=make_cfg(ckpt_every=3), source=ListSource(10), total_steps=6,
                         grad_accum=2)  # fmt: skip
    again.fit()
    assert file_steps(run) == [1, 2, 3, 4, 5, 6]
    assert len((run / "metrics_discarded.jsonl").read_text().splitlines()) == 9


def test_resume_truncates_lines_written_after_the_last_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    t = make_trainer(run, on_step=kill_at(5))  # ckpt at 3; lines for steps 4 and 5 are on disk
    with pytest.raises(Kill):
        t.fit()
    assert file_steps(run) == [1, 2, 3, 4, 5]
    t2 = make_trainer(run, model=build_model())
    assert t2.resume() is True
    assert file_steps(run) == [1, 2, 3]  # truncated to the checkpoint step before any new work
    t2.fit()
    assert file_steps(run) == list(range(1, 7))


# --------------------------------------------------------------------------------------------
# Eval loss: curves only, never a control signal
# --------------------------------------------------------------------------------------------

#: functions whose control flow decides when training stops, what is saved or restored, or what
#: the smoke gate says: none may mention a held-out loss in a condition
CONTROL_FUNCTIONS = (
    "fit", "resume", "save", "_optimizer_step", "save_checkpoint", "latest_checkpoint",
    "smoke_checks", "run_smoke", "lr_scale", "step_indices", "steps_per_epoch",
)  # fmt: skip


def _names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    return out


def test_no_code_path_reads_eval_loss_to_stop_or_to_select_a_checkpoint() -> None:
    tree = ast.parse(inspect.getsource(tr))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert set(CONTROL_FUNCTIONS) <= set(funcs)
    for name in CONTROL_FUNCTIONS:
        for node in ast.walk(funcs[name]):
            tests: list[ast.AST] = []
            if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
                tests.append(node.test)
            if isinstance(node, (ast.Break, ast.Raise)):
                pass  # a bare break / raise has no condition; the enclosing `if` is checked above
            if isinstance(node, ast.comprehension):
                tests += node.ifs
            for t in tests:
                bad = {x for x in _names(t) if "eval" in x.lower()}
                assert not bad, f"{name}: condition reads {bad}"
        assert not any(isinstance(n, ast.Break) for n in ast.walk(funcs[name])), name
    # nothing outside the record-keeping reads the value either
    assert "eval_loss" not in inspect.getsource(tr.smoke_checks)
    assert "eval" not in inspect.signature(tr.save_checkpoint).parameters
    assert "eval" not in "".join(inspect.signature(tr.latest_checkpoint).parameters)
    readers = [n for n, f in funcs.items() if "eval_loss" in {*_names(f)} and n not in (
        "_record_eval", "evaluate", "make_logger", "__init__")]  # fmt: skip
    assert readers == [], readers


@pytest.mark.parametrize("garbage", ["nan", "huge", "rising"])
def test_whatever_the_eval_loss_does_training_and_checkpoints_are_unchanged(
    tmp_path: Path, garbage: str
) -> None:
    ref = make_trainer(tmp_path / "ref", eval_source=None)  # no evaluation at all
    ref.fit()
    t = make_trainer(tmp_path / "t")
    seq = iter(range(1, 100))
    values = {
        "nan": lambda: float("nan"),
        "huge": lambda: 1e30,
        "rising": lambda: 10.0 ** next(seq),
    }
    t.evaluate = values[garbage]  # type: ignore[method-assign]
    t.fit()
    assert_same(snapshot(ref), snapshot(t))
    assert [r["loss"] for r in t.history] == [r["loss"] for r in ref.history]
    assert len(t.history) == t.total
    kept = lambda d: sorted(p.name for p in (d / "ckpt").iterdir())  # noqa: E731
    assert kept(tmp_path / "t") == kept(tmp_path / "ref")
    assert (tmp_path / "t" / "ckpt" / "LATEST").read_text() == "step_000006"  # last, not "best"
    assert any("eval_loss" in r for r in t.history) and not any(
        "eval_loss" in r for r in ref.history
    )


def test_evaluation_does_not_consume_training_rng(tmp_path: Path) -> None:
    """Dropout is off in eval mode, but the guarantee is explicit: states are restored."""
    t = make_trainer(tmp_path)
    before = tr.rng_state("cpu")
    t.model.train()
    t.evaluate()
    after = tr.rng_state("cpu")
    assert before["python"] == after["python"] and torch.equal(before["torch"], after["torch"])
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert t.model.training  # and the model is back in train mode


def test_epoch_end_eval_can_be_switched_off(tmp_path: Path) -> None:
    t = make_trainer(tmp_path, cfg=make_cfg(eval_every=0), eval_each_epoch=False)
    t.fit()
    assert not any("eval_loss" in r for r in t.history)
    t2 = make_trainer(tmp_path / "b", cfg=make_cfg(eval_every=0))  # epoch ends only: steps 3, 6
    t2.fit()
    assert [r["step"] for r in t2.history if "eval_loss" in r] == [3, 6]
    assert all("eval_loss_epoch" in r for r in t2.history if "eval_loss" in r)


def test_eval_subset_is_fixed_bounded_and_independent_of_the_epoch() -> None:
    specs = [ts.PageSpec(f"train_{i:04d}", "train", 0, 1, f"{i}.png") for i in range(230)]
    a = tr.eval_subset(specs, seed=42, n=24)
    assert len(a) == 24 and a == tr.eval_subset(specs, seed=42, n=24)
    assert len(set(a)) == 24 and set(a) <= set(specs)
    assert tr.eval_subset(specs[:10], seed=42, n=24) != []  # fewer pages than N: all of them
    assert len(tr.eval_subset(specs[:10], seed=42, n=24)) == 10
    assert a != tr.eval_subset(specs, seed=43, n=24)
