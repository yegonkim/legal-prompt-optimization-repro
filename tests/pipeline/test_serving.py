import copy
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from legal_prompt_optimization.pipeline.config import DEFAULTS
from legal_prompt_optimization.pipeline.serve import command, compatible, environment
from legal_prompt_optimization.pipeline.smoke import smoke


class Serving(unittest.TestCase):
    def test_single_console_command(self):
        project = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(set(project["project"]["scripts"]), {"legal-prompts"})

    def test_api_version_mismatches_fail_before_model_start(self):
        c = copy.deepcopy(DEFAULTS)
        compatible(c, "student", {"vllm": "0.33.0"})
        compatible(c, "proposer", {"vllm": "0.33.0"})
        with self.assertRaisesRegex(ValueError, "modern"):
            compatible(c, "student", {"vllm": "0.10.0"})
        with self.assertRaisesRegex(ValueError, "Gemma"):
            compatible(c, "proposer", {"vllm": "0.10.0"})
        c["generation"]["schema_mode"] = "guided_json"
        with self.assertRaisesRegex(ValueError, "guided"):
            compatible(c, "student", {"vllm": "0.33.0"})
        with self.assertRaisesRegex(RuntimeError, "--python"):
            environment("/nonexistent/vllm/python")

    def test_launcher_selects_api_flags_and_language_only_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(DEFAULTS)
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(c))
            args = SimpleNamespace(
                config=path,
                role="student",
                gpu=0,
                python="python",
                memory_fraction=0.4,
                max_sequences=8,
                phi_eos_alignment=False,
            )
            cmd, env = command(args)
            self.assertIn("--structured-outputs-config", cmd)
            self.assertNotIn("--enforce-eager", cmd)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            args.role = "proposer"
            cmd, _ = command(args)
            self.assertIn("--language-model-only", cmd)
            self.assertIn("gemma4", cmd)
            args.role = "student"
            c["models"]["student"]["api_key_env"] = "MODEL_KEY"
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, "authenticated"):
                command(args)

    def test_smoke_exercises_forcing_and_reports_failure(self):
        calls = []

        class Fake:
            def __init__(self, config, cache):
                self.config = config

            def preflight(self, **kwargs):
                return {"healthy": True}

            def generate(self, rows, instruction, seeds, judge=False):
                role = "judge" if judge else "response"
                cap = self.config["generation"][role]["max_tokens"]
                calls.append((rows[0]["task_type"], judge, cap))
                return [
                    {
                        "valid": True,
                        "text": '{"reasoning":"ok","answer":"Yes"}',
                        "answer_forcing": {"recovered": cap == 1},
                    }
                    for row in rows
                ]

            def reflect(self, prompt, **kwargs):
                return "```instruction```"

        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(DEFAULTS)
            original = copy.deepcopy(config)
            result = smoke(config, Path(tmp) / "pass", Fake)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(config, original)
            self.assertEqual(len([call for call in calls if call[2] == 1]), 4)
            self.assertIn(("regression", True, 1), calls)
            with (
                patch.object(Fake, "generate", return_value=[{"valid": False}]),
                self.assertRaisesRegex(RuntimeError, "valid outputs"),
            ):
                smoke(config, Path(tmp) / "fail", Fake)
            report = json.loads((Path(tmp) / "fail/smoke.json").read_text())
            self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
