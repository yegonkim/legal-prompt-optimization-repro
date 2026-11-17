"""CPU contract tests. Actual GEPA smoke runs only when dependency is installed."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from legal_prompt_optimization.simple_experiment import (
    TaskSpec, SearchConfig, Journal, CohortAdapter, metrics, evaluate_program,
    run_experiment, report_test, optimize,
)


TASK = TaskSpec(labels=("A", "B"))


def rows(prefix, answers=("A", "B")):
    return [dict(record_id=f"{prefix}{i}", question=f"choose {answer}", answer=answer,
                 reference_feedback="SECRET feedback", public_labels=["A", "B"])
            for i, answer in enumerate(answers)]


def program(prompt="ordinary"):
    return {**{f"actor{i}": prompt for i in range(4)}, "judge": prompt}


class Backend:
    def __init__(self):
        self.calls = []
        self.reflections = []

    def generate(self, rows, instructions, seeds, judge=False):
        self.calls.append((rows, instructions, seeds, judge))
        return [dict(prediction=r["question"].split()[-1], text=f"complete reasoning {s}", raw=[1, 2])
                for r, s in zip(rows, seeds)]

    def reflect(self, prompt):
        self.reflections.append(prompt)
        return "```Improved reusable instruction```"


class EvaluationBatch:
    def __init__(self, outputs, scores, trajectories):
        self.outputs, self.scores, self.trajectories = outputs, scores, trajectories


class FakeGEPA:
    """Contract fixture, explicitly not a replacement implementation of GEPA."""
    def __init__(self):
        self.seen = []

    def optimize(self, **kwargs):
        self.seen.append(kwargs)
        adapter = kwargs["adapter"]
        self_check = kwargs["valset"]
        assert len(self_check) == 1 and self_check[0]["split"] == "select"
        train_eval = adapter.evaluate(kwargs["trainset"][:1], kwargs["seed_candidate"], True)
        adapter.make_reflective_dataset(kwargs["seed_candidate"], train_eval, list(kwargs["seed_candidate"]))
        evaluated = adapter.evaluate(kwargs["valset"], kwargs["seed_candidate"])
        return types.SimpleNamespace(best_candidate=kwargs["seed_candidate"], best_idx=0,
                                     val_aggregate_scores=evaluated.scores)


def fake_modules(fake):
    return {"gepa": types.SimpleNamespace(optimize=fake.optimize),
            "gepa.core.adapter": types.SimpleNamespace(EvaluationBatch=EvaluationBatch)}


class SimpleExperimentTests(unittest.TestCase):
    def test_metrics_are_full_cohort(self):
        # Majority-only has higher accuracy but worse macro-F1 than minority-aware.
        gold = ["A"] * 8 + ["B"] * 2
        majority = metrics(TASK, gold, ["A"] * 10)
        minority = metrics(TASK, gold, ["A"] * 5 + ["B"] * 5)
        self.assertGreater(majority["accuracy"], minority["accuracy"])
        self.assertLess(majority["primary"], minority["primary"])
        correct = metrics(TASK, ["A", "B"], ["A", "B"])["primary"]
        singleton_mean = sum(metrics(TASK, [x], [x])["primary"] for x in ["A", "B"]) / 2
        self.assertEqual(correct, 1)
        self.assertEqual(singleton_mean, .5)

    def test_numeric_tasks(self):
        self.assertEqual(metrics(TaskSpec("regression"), [1, 3], [2, 5]), dict(primary=-1.5, mae=1.5))
        self.assertEqual(metrics(TaskSpec("probability"), [0, 1], [.5, .5]), dict(primary=.75, brier=.25))
        for prediction in (float("nan"), None, 2):
            with self.assertRaises(ValueError):
                metrics(TaskSpec("probability"), [0], [prediction])

    def test_exactly_five_calls_full_records_no_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = Backend()
            result = evaluate_program(backend, rows("dev"), program(), TASK, 42, Path(tmp) / "eval")
            self.assertEqual(result["calls"], 10)
            self.assertEqual(len(backend.calls), 5)
            self.assertEqual(result["judge"]["primary"], 1)
            actor_seeds = [call[2][0] for call in backend.calls[:4]]
            self.assertEqual(len(set(actor_seeds)), 4)
            for batch, _, _, _ in backend.calls:
                for row in batch:
                    self.assertNotIn("answer", row)
                    self.assertNotIn("reference_feedback", row)
            judges = backend.calls[-1]
            self.assertTrue(judges[3])
            self.assertEqual(len(judges[0][0]["candidates"]), 4)
            self.assertEqual(judges[0][0]["candidates"][0]["raw"], [1, 2])

    def test_adapter_selects_using_cohort_score_and_fit_only_reflection(self):
        class RankingBackend(Backend):
            def generate(self, batch, instructions, seeds, judge=False):
                return [dict(prediction=("A" if int(r["record_id"][1:]) < (10 if p == "majority" else 5) else "B"), text="reason")
                        for r, p in zip(batch, instructions)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, fake_modules(FakeGEPA())):
            data = rows("s", ["A"] * 8 + ["B"] * 2)
            adapter = CohortAdapter(RankingBackend(), TASK, 42, Journal(Path(tmp) / "log"), rows("f"), 1)
            scores = [adapter.evaluate([dict(split="select", rows=data)], {"response": p}, True)
                      for p in ("majority", "minority")]
            self.assertGreater(scores[1].scores[0], scores[0].scores[0])
            with self.assertRaisesRegex(ValueError, "FIT"):
                adapter.make_reflective_dataset({}, scores[0], ["response"])

    def test_frozen_judge_bank_and_joint_components(self):
        fake = FakeGEPA()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, fake_modules(fake)):
            b = Backend()
            result = run_experiment(b, {s: rows(s) for s in ("fit", "select", "dev")}, TASK,
                                    "ordinary", "judge", [SearchConfig("naive"), SearchConfig("grid")], Path(tmp) / "run")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["entries"]), 5)
            judge_search = fake.seen[1]
            self.assertEqual(judge_search["seed_candidate"], {"judge": "judge"})
            self.assertEqual(len(judge_search["trainset"][0]["rows"][0]["candidates"]), 4)
            self.assertEqual(set(fake.seen[2]["seed_candidate"]), {"actor0", "actor1", "actor2", "actor3", "judge"})
            selection = json.loads((Path(tmp) / "run/selection.json").read_text())
            report = report_test(b, rows("test"), TASK, selection, Path(tmp) / "test")
            self.assertEqual(report["calls"], 30)
            self.assertEqual(set(report["comparisons"]), {"baseline", "sequential", "joint"})
            self.assertTrue(report["reporting_only"])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                report_test(b, rows("dev"), TASK, selection, Path(tmp) / "bad")
            selection["program"]["judge"] = "tamper"
            with self.assertRaisesRegex(ValueError, "frozen"):
                report_test(b, rows("test"), TASK, selection, Path(tmp) / "bad")

    def test_split_boundaries_before_inference(self):
        b = Backend()
        splits = {s: rows(s) for s in ("fit", "select", "dev", "test")}
        with self.assertRaises(ValueError):
            run_experiment(b, splits, TASK, "a", "j", [SearchConfig("naive")], "unused")
        del splits["test"]
        splits["select"] = splits["fit"]
        with self.assertRaises(ValueError):
            run_experiment(b, splits, TASK, "a", "j", [SearchConfig("naive")], "unused")
        self.assertEqual(b.calls, [])

    def test_failed_raw_outputs_preserved_no_restart(self):
        class Broken(Backend):
            def generate(self, *args, **kwargs):
                return [dict(text="malformed raw output")]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "failure"
            with self.assertRaises(ValueError):
                evaluate_program(Broken(), rows("d"), program(), TASK, 42, directory)
            self.assertIn("malformed raw output", (directory / "events.jsonl").read_text())
            self.assertTrue((directory / "failure.json").exists())
            with self.assertRaises(FileExistsError):
                evaluate_program(Backend(), rows("d"), program(), TASK, 42, directory)

    def test_failures_do_not_claim_successful_optimization(self):
        fake = FakeGEPA()
        def fail(**kwargs):
            raise RuntimeError("synthetic optimizer failure")
        fake.optimize = fail
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, fake_modules(fake)):
            result = run_experiment(Backend(), {s: rows(s) for s in ("fit", "select", "dev")}, TASK,
                                    "ordinary", "judge", [SearchConfig("naive")], Path(tmp) / "run")
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(len(result["failures"]), 2)
            self.assertEqual(result["selection"]["method"], "baseline")


@unittest.skipUnless(importlib.util.find_spec("gepa"), "GEPA unavailable; no install performed")
class ActualGEPAIntegration(unittest.TestCase):
    def test_actual_select_prefers_macro_f1_over_accuracy(self):
        class MinorityAware(Backend):
            def generate(self, batch, instructions, seeds, judge=False):
                return [dict(prediction="A" if int(r["record_id"][1:]) < (5 if "Improved" in p else 10) else "B", text="reason")
                        for r, p in zip(batch, instructions)]
        with tempfile.TemporaryDirectory() as tmp:
            cohort_answers = ["A"] * 8 + ["B"] * 2
            winner = optimize(MinorityAware(), rows("f", cohort_answers), rows("s", cohort_answers),
                              TASK, {"response": "majority"}, SearchConfig("naive", fit_batch_size=10),
                              1, Path(tmp) / "actual")
            self.assertIn("Improved", winner["response"])
            record = json.loads((Path(tmp) / "actual/winner.json").read_text())
            self.assertGreater(record["scores"][1], record["scores"][0])

    def test_actual_cohort_optimizer(self):
        class Improving(Backend):
            def generate(self, batch, instructions, seeds, judge=False):
                return [dict(prediction=r["question"].split()[-1] if "Improved" in p else "A", text="reason")
                        for r, p in zip(batch, instructions)]
        with tempfile.TemporaryDirectory() as tmp:
            winner = optimize(Improving(), rows("f"), rows("s"), TASK, {"response": "ordinary"},
                              SearchConfig("naive", fit_batch_size=2), 1, Path(tmp) / "actual")
            self.assertIn("Improved", winner["response"])

    def test_actual_all_components_and_frozen_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = Backend()
            result = run_experiment(backend, {s: rows(s) for s in ("fit", "select", "dev")},
                                    TASK, "ordinary", "judge",
                                    [SearchConfig("naive", response_budget=1, judge_budget=1, joint_budget=5)],
                                    Path(tmp) / "run")
            self.assertEqual(result["status"], "complete")
            events = [json.loads(line) for line in (Path(tmp) / "run/naive/joint/events.jsonl").read_text().splitlines()]
            components = [e["components"] for e in events if e["event"] == "reflection_examples"]
            self.assertEqual(components, [[f"actor{i}"] for i in range(4)] + [["judge"]])
            self.assertEqual(len(backend.reflections), 7)
            self.assertEqual(set(result["selection"]["comparisons"]), {"baseline", "sequential", "joint"})


if __name__ == "__main__":
    unittest.main()

class NumericBaselineReportingTest(unittest.TestCase):
    def test_median4_for_both_numeric_tasks(self):
        for kind, values, gold, expected in [
            ('regression', [0, 2, 4, 100], 3, 0),
            ('probability', [0, .2, .8, 1], 1, .25),
        ]:
            class NumericBackend:
                index = 0
                def generate(self, rows, instructions, seeds, judge=False):
                    value = .5 if judge else values[self.index]
                    if not judge: self.index += 1
                    return [dict(prediction=value,text=str(value)) for _ in rows]
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                result=evaluate_program(NumericBackend(),
                    [dict(record_id='numeric',question='Synthetic problem',answer=gold)],
                    program(),TaskSpec(kind),42,Path(tmp)/'evaluation')
                self.assertAlmostEqual(result['median4']['mae' if kind=='regression' else 'brier'],expected)
                self.assertEqual(result['response_baseline']['pass1'],result['actors'][0])
                self.assertEqual(result['response_baseline']['median4'],result['median4'])
