"""CPU-only tests. All snippets here are synthetic, not redistributed benchmark rows."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from legal_prompt_optimization import simple_data as d


def rows(n):
    return [dict(id=f'row:{i:04d}', question=f'Synthetic question {i}', answer='A',
                 public_labels=['A', 'B']) for i in range(n)]


class SplitsTest(unittest.TestCase):
    def test_deterministic_disjoint_exhaustive_and_order_independent(self):
        source = rows(1400)
        a = d.split_rows(source)
        self.assertEqual(a, d.split_rows(list(reversed(source))))
        self.assertEqual({k: len(v) for k, v in a.items()},
                         dict(train=300, select=76, dev=512, test=512))
        ids = [r['id'] for part in a.values() for r in part]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {r['id'] for r in source})
        self.assertEqual(source, rows(1400))
        self.assertNotEqual(a, d.split_rows(source, seed=43))

    def test_small_source_explicit_sizes(self):
        with self.assertRaisesRegex(ValueError, 'explicit smaller'):
            d.split_rows(rows(387))
        a = d.split_rows(rows(387), dev_size=64, test_size=64, select_size=52)
        self.assertEqual({k: len(v) for k, v in a.items()},
                         dict(train=207, select=52, dev=64, test=64))

    def test_invalid_sizes_duplicates_and_empty_sources(self):
        for kw in ({'dev_size': 513}, {'test_size': 0}, {'dev_size': True},
                   {'select_size': -1}, {'select_size': 376}, {'select_size': 3.5}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                d.split_rows(rows(1400), **kw)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            d.split_rows(rows(1400) + rows(1))
        with self.assertRaises(ValueError):
            d.split_rows([])


class SchemaTest(unittest.TestCase):
    def test_numeric_and_classification_validation(self):
        for task, answer in [('regression', 0), ('regression', 120.5), ('probability', 1)]:
            d.validate_rows([dict(id='x', question='Q', task_type=task, answer=answer)])
        for answer in [float('nan'), float('inf'), True, 'unknown', '10']:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                d.validate_rows([dict(id='x', question='Q', task_type='regression', answer=answer)])
        for answer in [-.1, 1.1]:
            with self.assertRaises(ValueError):
                d.validate_rows([dict(id='x', question='Q', task_type='probability', answer=answer)])
        for labels in [['A', 'A'], ['A'], ['A', ''], 'AB']:
            with self.assertRaises(ValueError):
                d.validate_rows([dict(id='x', question='Q', answer='A', public_labels=labels)])

    def test_npd_never_exposes_targets_or_beneficiaries(self):
        raw = {key: 'unknown' for key in d.NPD_INPUTS}
        raw.update(case_id='case-1', y_amount_eur='12500', y_binary='1',
                   target_status='TARGET_SECRET', beneficiary_type='BENEFICIARY_SECRET',
                   has_joint_beneficiary='ALLOCATION_SECRET', split='test',
                   arbitrary_new_column='SECRET_NEW_FIELD')
        result = d.normalize_npd([raw])[0]
        self.assertEqual(result['answer'], 12500.)
        self.assertEqual(result['task_type'], 'regression')
        for secret in ['12500', 'TARGET_SECRET', 'BENEFICIARY_SECRET', 'ALLOCATION_SECRET', 'SECRET_NEW_FIELD']:
            self.assertNotIn(secret, result['question'])
        raw['y_amount_eur'] = 'unknown'
        with self.assertRaises(ValueError):
            d.normalize_npd([raw])

    def test_parcel_requires_premise_and_excludes_gold_reasoning(self):
        raw = [dict(claim_id=1, citation='Synthetic citation', input_claim='A claim',
                    ground_truth_label='Refuted', ground_truth_reasoning='GOLD_SECRET')]
        with self.assertRaises(d.SourceUnavailableError):
            d.normalize_parcel(raw, {})
        result = d.normalize_parcel(raw, {'Synthetic citation': 'Full synthetic opinion'})
        d.validate_rows(result)
        self.assertIn('Full synthetic opinion', result[0]['question'])
        self.assertNotIn('GOLD_SECRET', result[0]['question'])
        self.assertEqual(result[0]['answer'], 'Refuted')

    def test_sara_entailment_only_with_complete_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sara.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                entries = {'sara/cases/s1_test.pl': '% Text\n% Synthetic facts\n% Question\n% A statement Entailment\n% Facts\n',
                           'sara/cases/s1_numeric.pl': '% Text\n% Facts\n% Question\n% Amount 100\n% Facts\n',
                           'sara/statutes/source/section1': 'Complete synthetic statute\nFinal section clause'}
                for name, text in entries.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(text.encode())
                    archive.addfile(member, io.BytesIO(text.encode()))
            result = d._sara(path)
            self.assertEqual(len(result), 1)
            self.assertIn('Final section clause', result[0]['question'])
            self.assertNotIn('Entailment', result[0]['question'])


class IOTest(unittest.TestCase):
    def test_prepare_no_manifest_no_overwrite_and_no_mutation_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.jsonl'
            source.write_text(''.join(json.dumps(r) + '\n' for r in rows(20)))
            out = Path(tmp) / 'prepared'
            counts = d.prepare('casehold', out, source_path=source, dev_size=4, test_size=4, select_size=3)
            self.assertEqual(counts, dict(train=9, select=3, dev=4, test=4))
            self.assertEqual({p.name for p in out.iterdir()}, {'train.jsonl', 'select.jsonl', 'dev.jsonl', 'test.jsonl'})
            with self.assertRaises(FileExistsError):
                d.prepare('casehold', out, source_path=source)
            failed = Path(tmp) / 'failed'
            with self.assertRaises(ValueError):
                d.prepare('casehold', failed, source_path=source)
            self.assertFalse(failed.exists())

    def test_forecasts_are_not_gold_and_missing_packets_are_explicit(self):
        with self.assertRaisesRegex(d.SourceUnavailableError, 'input packets'):
            d.load_rows('legalforecastbench')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.jsonl'
            r = dict(id='unit1', question='Synthetic pre-decision filings', task_type='probability', answer=.97)
            path.write_text(json.dumps(r) + '\n')
            with self.assertRaisesRegex(ValueError, 'binary observed outcomes'):
                d.load_rows('legalforecastbench', source_path=path)
            r['answer'] = 1
            path.write_text(json.dumps(r) + '\n')
            self.assertEqual(d.load_rows('legalforecastbench', source_path=path), [r])

    def test_download_offline_atomic_failure_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            url = 'https://example.invalid/synthetic.csv'
            with self.assertRaises(d.SourceUnavailableError):
                d._download(url, tmp, offline=True)
            with patch.object(d.urllib.request, 'urlopen', side_effect=urllib.error.URLError('offline')):
                with self.assertRaises(urllib.error.URLError):
                    d._download(url, tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with patch.object(d.urllib.request, 'urlopen', return_value=io.BytesIO(b'fixture')):
                path = d._download(url, tmp)
            self.assertEqual(d._download(url, tmp, offline=True), path)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                d._download(url, tmp, offline=True, sha256='bad')

    def test_cli_interface(self):
        with patch.object(d, 'prepare', return_value={'train': 207}) as prepare, patch('sys.stdout', new_callable=io.StringIO):
            d.main(['--dataset', 'legalforecastbench', '--data', 'somewhere',
                    '--source-path', 'local.jsonl', '--dev-size', '64', '--test-size', '64', '--select-size', '52'])
        self.assertEqual(prepare.call_args.kwargs['dev_size'], 64)
        self.assertEqual(prepare.call_args.kwargs['out'], 'somewhere')


if __name__ == '__main__':
    unittest.main()

class LexamVariantsTest(unittest.TestCase):
    def test_same_question_variants_cannot_cross_splits(self):
        from types import SimpleNamespace
        row = dict(id='shared',language='en',question='Synthetic question',course='Law',
                   jurisdiction='Synthetic',choices=['one','two','three','four'],gold=0)
        table=SimpleNamespace(to_pylist=lambda:[row])
        with patch.object(d,'_download',return_value=Path('unused')), patch.object(d,'_parquet',return_value=table):
            result=d.load_rows('lexam')
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['id'],'lexam:shared')
