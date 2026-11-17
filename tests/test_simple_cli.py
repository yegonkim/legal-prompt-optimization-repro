import tempfile
from pathlib import Path
import unittest
from legal_prompt_optimization.simple_cli import read_rows
from legal_prompt_optimization.simple_backend import SimpleBackend

class IntegrationTests(unittest.TestCase):
    def test_rows_map_ids_and_limit_without_reading_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.jsonl'
            path.write_text('{"id":"first","question":"q"}\n{"id":"second","question":"q2"}\n')
            self.assertEqual([r['record_id'] for r in read_rows(path,1)],['first'])

    def test_call_budget_fails_before_network(self):
        backend=SimpleBackend('http://127.0.0.1:1',max_physical_calls=0)
        with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
            backend._request(backend.student_endpoint,'/v1/chat/completions',{})
        self.assertEqual(backend.physical_calls,0)
