"""Selection-rule regressions; no downloaded models, datasets or case-ID lists."""

import unittest
from pathlib import Path
from unittest.mock import patch

from legal_prompt_optimization.data import cohort_digest
from legal_prompt_optimization.pipeline.io import validate_contract
from legal_prompt_optimization.pipeline.recipes import (
    build,
    casehold_eligible,
    protected_ledgar_ids,
    reporting_halves,
    shuffled,
)


def rows(n, prefix="id"):
    return [
        {
            "record_id": f"{prefix}:{i:06d}",
            "question": str(i),
            "answer": "A",
            "public_labels": ["A", "B"],
        }
        for i in range(n)
    ]


class Recipes(unittest.TestCase):
    def test_reporting_half_cap_does_not_resplit_or_move_test_boundary(self):
        pool = rows(3000)
        whole = shuffled(pool)
        dev, test = reporting_halves(pool)
        self.assertEqual(len(dev), 512)
        self.assertEqual(len(test), 512)
        self.assertEqual(dev, whole[:512])
        self.assertEqual(test, whole[1024:1536])
        small = rows(101)
        dev, test = reporting_halves(small)
        self.assertEqual((len(dev), len(test)), (50, 51))
        self.assertEqual(test[0], shuffled(small)[50])

    def test_ledgar_exclusion_schedule_recreated_without_saved_ids(self):
        pool = rows(10000, "ledgar:test")
        protected = protected_ledgar_ids(pool)
        self.assertEqual(len(protected), 256)
        self.assertFalse(protected & {r["record_id"] for r in shuffled(pool)[:256]})
        _, test = reporting_halves([r for r in pool if r["record_id"] not in protected])
        self.assertEqual(test[0]["record_id"], "ledgar:test:006176")

    def test_casehold_filter_border_and_special_token_mode(self):
        class Tokenizer:
            def __call__(self, texts, *, add_special_tokens, truncation, return_length):
                assert truncation is False
                return {"length": [int(x) + int(add_special_tokens) for x in texts]}

        pool = [
            {"record_id": str(i), "question": str(n)}
            for i, n in enumerate([1799, 1800, 1801])
        ]
        self.assertEqual(
            len(
                casehold_eligible(
                    pool, [Tokenizer(), Tokenizer()], add_special_tokens=True
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                casehold_eligible(
                    pool, [Tokenizer(), Tokenizer()], add_special_tokens=False
                )
            ),
            2,
        )

    def test_lexam_excludes_entire_prior_research_pool(self):
        pool = rows(1807)
        prior = {r["record_id"] for r in shuffled(pool)[:320]}
        with (
            patch(
                "legal_prompt_optimization.pipeline.recipes.read_sources",
                return_value={},
            ),
            patch(
                "legal_prompt_optimization.pipeline.recipes.lexam_rows",
                return_value=pool,
            ),
        ):
            groups, details = build("lexam", Path("unused"), offline=True)
        self.assertEqual(details["research_exclusion_count"], 320)
        self.assertTrue(
            all(r["record_id"] not in prior for s in ["dev", "test"] for r in groups[s])
        )
        self.assertEqual((len(groups["dev"]), len(groups["test"])), (512, 512))

    def test_split_fingerprint_rejects_reordering_and_content_changes(self):
        pool = rows(3)
        contract = {
            "dataset": "toy",
            "expected_counts": {"fit": 3},
            "split_sha256": {"fit": cohort_digest(pool)},
        }
        validate_contract({"fit": pool}, contract)
        with self.assertRaises(ValueError):
            validate_contract({"fit": pool[::-1]}, contract)
        changed = [dict(r) for r in pool]
        changed[0]["question"] = "changed"
        with self.assertRaises(ValueError):
            validate_contract({"fit": changed}, contract)
