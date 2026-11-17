import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import yaml

from legal_prompt_optimization.core import OUTPUT_INSTRUCTION, seed_for
from legal_prompt_optimization.pipeline.backend import Backend
from legal_prompt_optimization.pipeline.cli import main
from legal_prompt_optimization.pipeline.config import load
from legal_prompt_optimization.pipeline.historical import stage_config

REFERENCE = Path("configs/pipeline/ledgar_historical.yaml")


class FakeEndpoint:
    calls: ClassVar[list] = []
    reflections: ClassVar[list] = []

    def __init__(self, config, cache):
        self.config = config

    def preflight(self, **kwargs):
        return {"ok": True}

    def generate(self, rows, instructions, seeds, judge=False):
        if isinstance(instructions, str):
            instructions = [instructions] * len(rows)
        self.calls.append((rows, instructions, seeds, judge))
        results = []
        for row, instruction, seed in zip(rows, instructions, seeds):
            assert "answer" not in row and "reference_feedback" not in row
            answer = "B" if "select B" in row["question"] else "A"
            if not instruction.endswith("improved reusable instruction"):
                answer = "A"
            results.append(
                {
                    "text": json.dumps({"reasoning": "analysis", "answer": answer}),
                    "prediction": answer,
                    "valid": True,
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 10},
                    "seed": seed,
                }
            )
        return results

    def reflect(self, prompt, *, call_number=None):
        assert "test0" not in prompt and "dev0" not in prompt
        self.reflections.append((call_number, prompt))
        return "```improved reusable instruction```"


class HistoricalProtocol(unittest.TestCase):
    def test_effective_reference_settings(self):
        config = load(REFERENCE)
        self.assertEqual(config["gepa"]["judge"]["reflection_batch_size"], 16)
        self.assertEqual(config["gepa"]["judge"]["max_proposals"], 64)
        self.assertEqual(
            stage_config(config, "judge")["student_call_cap"] - 4 * (128 + 64), 7643
        )
        self.assertEqual(stage_config(config, "response")["seed"], 42)
        self.assertEqual(stage_config(config, "judge")["seed"], 43)
        self.assertFalse(config["generation"]["answer_forcing"]["enabled"])

    def test_actual_gepa_per_case_separate_seeds_prefix_and_16_draw_readout(self):
        import gepa

        FakeEndpoint.calls = []
        FakeEndpoint.reflections = []
        seen = []
        real = gepa.optimize

        def capture(**kwargs):
            seen.append(kwargs)
            return real(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load(REFERENCE)
            config["models"]["student"]["tokenizer"] = None
            config["data"] = {
                "dataset": None,
                "split_sha256": dict.fromkeys(["fit", "select", "dev", "test"]),
                "expected_counts": dict.fromkeys(["fit", "select", "dev", "test"]),
            }
            for stage in ["response", "judge"]:
                config["gepa"][stage]["max_proposals"] = 1
                config["gepa"][stage]["max_solver_calls"] = 100
                config["gepa"][stage]["max_metric_calls"] = 100
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            rows = [
                json.loads(line)
                for line in Path("examples/toy.jsonl").read_text().splitlines()
            ]
            for row in rows:
                if row["split"] == "fit":
                    row["reference_feedback"] = "FIT-only feedback: " + row["answer"]
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps(r) for r in rows))
            out = root / "opt"
            with patch("gepa.optimize", side_effect=capture):
                result = main(
                    [
                        "optimize",
                        "--config",
                        str(path),
                        "--data",
                        str(data),
                        "--out",
                        str(out),
                    ],
                    backend_factory=FakeEndpoint,
                )
            self.assertEqual([x["seed"] for x in seen], [42, 43])
            self.assertEqual([x["reflection_minibatch_size"] for x in seen], [3, 16])
            self.assertEqual(len(seen[0]["valset"]), 2)
            self.assertTrue(all("rows" not in r for r in seen[0]["valset"]))
            self.assertEqual(
                seen[1]["max_metric_calls"], 84
            )  # 16 candidate-bank draws charged.
            self.assertTrue(
                all(
                    not row["record_id"].startswith(("dev", "test"))
                    for call in FakeEndpoint.calls
                    for row in call[0]
                )
            )
            self.assertTrue(
                any(
                    judge
                    and all(
                        p.startswith(result["program"]["response"] + "\n\n")
                        for p in prompts
                    )
                    for _, prompts, _, judge in FakeEndpoint.calls
                )
            )
            self.assertEqual(
                FakeEndpoint.calls[0][2],
                [seed_for("select0", 0), seed_for("select1", 0)],
            )
            count = sum(len(call[0]) for call in FakeEndpoint.calls)
            report = main(
                [
                    "evaluate",
                    "--config",
                    str(path),
                    "--data",
                    str(data),
                    "--winner",
                    str(out / "winner.json"),
                    "--split",
                    "test",
                    "--out",
                    str(root / "eval"),
                ],
                backend_factory=FakeEndpoint,
            )
            self.assertEqual(
                sum(len(call[0]) for call in FakeEndpoint.calls) - count, 72
            )
            self.assertEqual(report["baseline"]["pass1"]["n"], 32)
            self.assertEqual(report["optimized_response"]["vote4"]["n"], 8)
            self.assertEqual(report["judge"]["n"], 8)
            self.assertEqual(report["baseline"]["pass1"]["accuracy"], 0.5)
            self.assertEqual(report["optimized_response"]["pass1"]["accuracy"], 1)
            self.assertFalse(report["historical_gpu_score_reproduction_validated"])

    def test_reference_contract_rejects_wrong_cohort_before_generation(self):
        from legal_prompt_optimization.pipeline.io import dataset, validate_contract

        groups = dataset("examples/toy.jsonl")
        with self.assertRaisesRegex(ValueError, "expected 128"):
            validate_contract(groups, load(REFERENCE)["data"])

    def test_historical_protocol_rejects_forcing(self):
        from legal_prompt_optimization.pipeline.config import validate

        config = load(REFERENCE)
        config["generation"]["answer_forcing"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "no answer forcing"):
            validate(config)

    @unittest.skipUnless(
        importlib.util.find_spec("xgrammar"), "optional grammar dependency"
    )
    def test_native_contract_compact_grammar_and_seeded_reflector(self):
        from legal_prompt_optimization.simple_backend import SimpleBackend

        calls = []

        def request(backend, endpoint, path, payload=None, **kwargs):
            calls.append(payload)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"reasoning":"ok","answer":"A"}'},
                    }
                ],
                "usage": {"completion_tokens": 10},
            }

        with tempfile.TemporaryDirectory() as tmp:
            config = load(REFERENCE)
            config["models"]["student"]["tokenizer"] = None
            backend = Backend(config, tmp)
            backend._count = lambda *a, **k: 100
            with patch.object(SimpleBackend, "_request", request):
                backend.generate(
                    [
                        {
                            "record_id": "x",
                            "question": "original question",
                            "task_type": "classification",
                            "public_labels": ["A", "B"],
                        }
                    ],
                    "fixed prefix\n\njudge",
                    [42],
                    judge=True,
                )
                backend.reflect("FIT feedback", call_number=1)
            self.assertEqual(
                calls[0]["messages"],
                [
                    {
                        "role": "system",
                        "content": "fixed prefix\n\njudge\n\n" + OUTPUT_INSTRUCTION,
                    },
                    {"role": "user", "content": "original question"},
                ],
            )
            self.assertIn("guided_grammar", calls[0])
            self.assertNotIn("guided_json", calls[0])
            self.assertEqual(calls[0]["temperature"], 0)
            self.assertEqual(calls[0]["max_tokens"], 512)
            self.assertEqual(calls[1]["seed"], 6001)
            self.assertEqual(calls[1]["top_k"], 20)
            self.assertTrue(calls[1]["chat_template_kwargs"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
