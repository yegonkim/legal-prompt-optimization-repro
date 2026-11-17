import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import yaml

from legal_prompt_optimization.pipeline.cli import main
from legal_prompt_optimization.pipeline.config import load
from legal_prompt_optimization.pipeline.engine import TaskSpec, aggregate, metric
from legal_prompt_optimization.pipeline.io import dataset, write


class FakeBackend:
    calls: ClassVar[list] = []
    prompts: ClassVar[list] = []

    def __init__(self, config, cache):
        pass

    def preflight(self, **kw):
        return {"ok": True}

    def generate(self, rows, instructions, seeds, judge=False):
        if isinstance(instructions, str):
            instructions = [instructions] * len(rows)
        self.calls.append((rows, instructions, seeds, judge))
        for r in rows:
            assert "answer" not in r and "reference_feedback" not in r
            if judge:
                assert len(r["candidates"]) == 4
        return [
            {
                "prediction": r["question"].split()[-1] if "improved" in p else "A",
                "valid": True,
                "text": "complete reasoning",
                "seed": s,
            }
            for r, p, s in zip(rows, instructions, seeds)
        ]

    def reflect(self, prompt):
        self.prompts.append(prompt)
        assert "test0" not in prompt and "dev0" not in prompt
        return "```improved reusable instruction```"


class Workflow(unittest.TestCase):
    def test_actual_gepa_optimize_evaluate_resume_and_tamper_guard(self):
        FakeBackend.calls = []
        FakeBackend.prompts = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "optimization"
            evaluation = root / "evaluation"
            config = root / "config.yaml"
            config.write_text(Path("examples/smoke.yaml").read_text())
            data = Path("examples/toy.jsonl")
            opt = [
                "optimize",
                "--config",
                str(config),
                "--data",
                str(data),
                "--out",
                str(out),
            ]
            result = main(opt, backend_factory=FakeBackend)
            self.assertEqual(set(result["program"]), {"response", "judge"})
            self.assertTrue(FakeBackend.prompts)
            self.assertTrue(
                all(
                    not r["record_id"].startswith(("test", "dev"))
                    for call in FakeBackend.calls
                    for r in call[0]
                )
            )
            count = len(FakeBackend.calls)
            main(opt + ["--resume"], backend_factory=FakeBackend)
            self.assertEqual(count, len(FakeBackend.calls))
            ev = [
                "evaluate",
                "--config",
                str(config),
                "--data",
                str(data),
                "--winner",
                str(out / "winner.json"),
                "--out",
                str(evaluation),
                "--split",
                "test",
            ]
            report = main(ev, backend_factory=FakeBackend)
            self.assertEqual(report["n"], 2)
            self.assertTrue((evaluation / "table.md").exists())
            calls = FakeBackend.calls[count:]
            self.assertEqual(sum(len(c[0]) for c in calls), 18)
            count = len(FakeBackend.calls)
            main(ev + ["--resume"], backend_factory=FakeBackend)
            self.assertEqual(count, len(FakeBackend.calls))
            original = json.loads((out / "winner.json").read_text())
            original["program"]["response"] = "tampered"
            write(out / "winner.json", original)
            with self.assertRaisesRegex(ValueError, "hash"):
                main(ev, backend_factory=FakeBackend)

    def test_response_judge_and_joint_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            settings = yaml.safe_load(Path("examples/smoke.yaml").read_text())
            for mode in ["response", "judge", "joint"]:
                settings["mode"] = mode
                config.write_text(yaml.safe_dump(settings))
                args = [
                    "optimize",
                    "--config",
                    str(config),
                    "--data",
                    "examples/toy.jsonl",
                    "--out",
                    str(root / mode),
                ]
                if mode == "judge":
                    args += ["--response", str(root / "response" / "winner.json")]
                result = main(args, backend_factory=FakeBackend)
                expected = (
                    {"response"}
                    if mode == "response"
                    else {"response", "judge"}
                    if mode == "judge"
                    else {"actor0", "actor1", "actor2", "actor3", "judge"}
                )
                self.assertEqual(set(result["program"]), expected)

    def test_unknown_config_keys_and_wrong_types_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.yaml"
            for cfg in [
                {"generation": {"answer_forcing": {"enabled": "false"}}},
                {"gepa": {"response": {"temperature": 0.7}}},
                {"runtime": {"concurrency": True}},
            ]:
                p.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    load(p)

    def test_overlapping_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {
                    "id": "same",
                    "split": s,
                    "question": "q",
                    "answer": "A",
                    "public_labels": ["A", "B"],
                }
                for s in ["fit", "test"]
            ]
            p = Path(tmp) / "data.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                dataset(p)

    def test_valid_only_numeric_metrics_and_remaining_draws(self):
        task = TaskSpec("regression")
        m = metric(task, [0, 2], [None, 4])
        self.assertEqual(m["mae"], 2)
        self.assertEqual(m["coverage"], 0.5)
        self.assertAlmostEqual(metric(task, [0, 0], [1e308, 1e308])["mae"], 1e308)
        self.assertFalse(metric(task, [0], [None])["selection_eligible"])
        self.assertEqual(
            aggregate([{"prediction": 1e308, "valid": True}] * 4, "regression"), 1e308
        )
        self.assertEqual(
            aggregate(
                [
                    {"prediction": 10, "valid": True},
                    {"prediction": None, "valid": False},
                ],
                "regression",
            ),
            10,
        )
        self.assertEqual(
            aggregate(
                [{"prediction": x, "valid": True} for x in ["B", "A", "A", "B"]],
                "classification",
            ),
            "B",
        )


if __name__ == "__main__":
    unittest.main()
