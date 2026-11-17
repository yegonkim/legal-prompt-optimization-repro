import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import escape

from legal_prompt_optimization.pipeline.io import dataset
from legal_prompt_optimization.pipeline.prepare import extract_facts, prepare
from legal_prompt_optimization.simple_data import NPD_INPUTS


class Tokenizer:
    def encode(self, text, **kw):
        return text.split()


def docx(facts):
    out = io.BytesIO()

    def p(text):
        return "<w:p><w:r><w:t>" + escape(text) + "</w:t></w:r></w:p>"

    body = (
        p("THE FACTS")
        + p(facts)
        + "<w:tbl><w:tr><w:tc>"
        + p("inline factual detail")
        + "</w:tc></w:tr></w:tbl>"
        + p("THE LAW")
        + p("APPLICATION OF ARTICLE 41")
        + p("SECRET_AWARD 999999")
    )
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            + body
            + "</w:body></w:document>",
        )
    return out.getvalue()


class Preparation(unittest.TestCase):
    def test_reference_builder_retains_source_identity_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data.jsonl"
            groups = {
                split: [
                    {
                        "record_id": split,
                        "source_record_id": "source:" + split,
                        "question": "Synthetic question",
                        "answer": "A",
                        "public_labels": ["A", "B"],
                        "reference_feedback": "FIT feedback",
                    }
                ]
                for split in ["fit", "select", "dev", "test"]
            }
            with patch(
                "legal_prompt_optimization.pipeline.recipes.build",
                return_value=(groups, {"recipe": "reference_v1"}),
            ):
                prepare(dataset="ledgar", out=out, offline=True)
            row = dataset(out)["fit"][0]
            self.assertEqual(row["source_record_id"], "source:" + row["record_id"])
            self.assertEqual(row["reference_feedback"], "FIT feedback")

    def test_facts_include_inline_table_but_not_awards(self):
        facts = extract_facts(docx("The applicant was detained."))
        self.assertIn("inline factual detail", facts)
        self.assertNotIn("SECRET_AWARD", facts)

    def test_npd_filters_full_official_pools_before_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            raw = []
            for i, split in enumerate(
                ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
            ):
                case = f"001-{i + 1}"
                raw.append(
                    {
                        **{k: "unknown" for k in NPD_INPUTS},
                        "case_id": case,
                        "split": split,
                        "y_amount_eur": "100",
                    }
                )
                (docs / (case + ".docx")).write_bytes(
                    docx("long " * 100 if i == 0 else "Permitted factual description.")
                )
            source = root / "source.csv"
            with source.open("w") as f:
                w = csv.DictWriter(f, fieldnames=list(raw[0]))
                w.writeheader()
                w.writerows(raw)
            with patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ):
                result = prepare(
                    "ecthr_npd",
                    root / "out.jsonl",
                    source=source,
                    judgments_dir=docs,
                    offline=True,
                    fit_size=1,
                    select_size=1,
                    dev_size=1,
                    test_size=1,
                    max_input_tokens=50,
                )
            groups = dataset(root / "out.jsonl")
            self.assertEqual(
                result["counts"], {"fit": 1, "select": 1, "dev": 1, "test": 1}
            )
            self.assertTrue(
                all(
                    r["id"].split(":")[-1] in {"001-2", "001-3", "001-4"}
                    for s in ["fit", "select"]
                    for r in groups[s]
                )
            )
            self.assertTrue(groups["dev"][0]["id"].endswith(("001-5", "001-6")))
            self.assertTrue(groups["test"][0]["id"].endswith(("001-7", "001-8")))
            self.assertNotIn("SECRET_AWARD", (root / "out.jsonl").read_text())

    def test_parcel_uses_full_opinions_and_reports_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            citation = "2025 NY Slip Op 00001"
            text = citation + " Court of Appeals\n" + "Full opinion. " * 200
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "claim_id": i,
                            "citation": citation,
                            "input_claim": "claim" + str(i),
                            "ground_truth_label": "Supported",
                        }
                    )
                    for i in range(4)
                )
            )
            opinions = root / "opinions.json"
            opinions.write_text(json.dumps({citation: text}))
            with patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ):
                prepare(
                    "parcel",
                    root / "out.jsonl",
                    source=source,
                    opinions=opinions,
                    offline=True,
                    fit_size=1,
                    select_size=1,
                    dev_size=1,
                    test_size=1,
                )
            groups = dataset(root / "out.jsonl")
            self.assertIn(text.strip(), groups["fit"][0]["question"])
            provenance = json.loads((root / "out.jsonl.provenance.json").read_text())
            self.assertEqual(provenance["opinion_overlap"]["dev__test"], 1)

    def test_random_builder_requires_enough_rows(self):
        from legal_prompt_optimization.pipeline.prepare import split_pool

        with self.assertRaisesRegex(ValueError, "eligible"):
            split_pool([{"id": "a"}], {"fit": 2}, 42)


if __name__ == "__main__":
    unittest.main()
