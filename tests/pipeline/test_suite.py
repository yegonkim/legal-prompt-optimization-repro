import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from legal_prompt_optimization.pipeline.config import load
from legal_prompt_optimization.pipeline.io import digest
from legal_prompt_optimization.pipeline.suite import commands, freeze_archived


class SuiteTests(unittest.TestCase):
    def test_all_configs_and_commands(self):
        root = Path.cwd()
        suite = yaml.safe_load(Path("configs/experiments/suite.yaml").read_text())
        names = list(suite["datasets"])
        self.assertEqual(len(names), 8)
        cmds = list(
            commands(
                suite,
                root,
                names,
                ["prepare", "optimize", "dev", "test"],
                Path("/tmp/unused-suite"),
            )
        )
        self.assertEqual(len(cmds), 32)
        for i in range(0, 32, 4):
            self.assertIn("prepare", cmds[i])
            self.assertIn("optimize", cmds[i + 1])
            self.assertIn("dev", cmds[i + 2])
            self.assertIn("test", cmds[i + 3])
        imports = list(
            commands(
                suite, root, ["ledgar"], ["optimize"], Path("/tmp/unused-suite"), True
            )
        )
        self.assertIn("freeze-archived", imports[0])
        self.assertNotIn("optimize", imports[0])
        self.assertEqual(load("configs/pipeline/parcel_joint.yaml")["mode"], "joint")

    def test_resume_only_when_output_exists_and_existing_data_is_validated(self):
        suite = yaml.safe_load(Path("configs/experiments/suite.yaml").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = list(commands(suite, Path.cwd(), ["ledgar"], ["optimize"], out))
            self.assertNotIn("--resume", first[0])
            (out / "ledgar/optimization").mkdir(parents=True)
            (out / "data").mkdir()
            (out / "data/ledgar.jsonl").write_text("existing")
            resumed = list(
                commands(suite, Path.cwd(), ["ledgar"], ["prepare", "optimize"], out)
            )
            self.assertIn("validate", resumed[0])
            self.assertIn("--resume", resumed[1])

    def test_archived_import_checks_hash_and_cohort(self):
        # No endpoint or optimizer: importing prompts must never generate predictions.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data = Path("examples/toy.jsonl")
            cfg = load("examples/smoke.yaml")
            cfg["data"]["manifest"] = "toy"
            cfg["data"]["manifest_sha256"] = "0" * 64
            cp = p / "config.yaml"
            cp.write_text(yaml.safe_dump(cfg))
            prompt = {
                "dataset": "toy",
                "program": {"response": "r", "judge": "j"},
                "sha256": {
                    k: hashlib.sha256(v.encode()).hexdigest()
                    for k, v in {"response": "r", "judge": "j"}.items()
                },
            }
            pp = p / "prompts.json"
            pp.write_text(json.dumps(prompt))
            from unittest.mock import patch

            with patch(
                "legal_prompt_optimization.pipeline.suite.validate_contract"
            ) as validate:
                result = freeze_archived(data, cp, pp, p / "winner.json")
                validate.assert_called_once()
                self.assertIn("no new optimization", result["selected_on"])
                self.assertEqual(
                    result["sha256"],
                    digest({k: v for k, v in result.items() if k != "sha256"}),
                )
                prompt["program"]["response"] = "tampered"
                pp.write_text(json.dumps(prompt))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    freeze_archived(data, cp, pp, p / "winner.json")
            with self.assertRaises((ValueError, FileNotFoundError)):
                freeze_archived(data, cp, pp, p / "winner.json")
