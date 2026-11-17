"""CPU-only endpoint contract tests: no vLLM/transformers imports or GPU launches."""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import URLError

from legal_prompt_optimization.simple_backend import SimpleBackend, ContextLimitError, answer_schema


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(''.join(m['content'] for m in messages))


def row(**extra):
    return dict(question='Choose A or B.', public_labels=['A', 'B'], **extra)


def completion(answer='A', finish='stop'):
    return {'id': 'test', 'usage': {'prompt_tokens': 12, 'completion_tokens': 9, 'total_tokens': 21},
            'choices': [{'index': 0, 'finish_reason': finish, 'message': {
                'content': json.dumps({'reasoning': 'Complete reasoning\nincluding "quotes".', 'answer': answer})}}]}


class BackendTests(unittest.TestCase):
    def backend(self, **kw):
        return SimpleBackend('http://student/v1', 'http://proposer/v1', proposer_model='gemma',
                             student_tokenizer=CharTokenizer(), proposer_tokenizer=CharTokenizer(), **kw)

    def test_modes_single_draw_and_ordered_concurrency(self):
        for mode in ('guided_json', 'structured_outputs'):
            b = self.backend(structured_output_mode=mode)
            calls = []
            def request(endpoint, path, payload):
                calls.append(payload)
                time.sleep(0.005 if payload['seed'] == 1 else 0)
                return completion('A' if payload['seed'] == 1 else 'B')
            with patch.object(b, '_request', side_effect=request):
                result = b.generate([row(), row()], ['one', 'two'], [1, 2])
            self.assertEqual([r['prediction'] for r in result], ['A', 'B'])
            self.assertTrue(all(c['n'] == 1 for c in calls))
            self.assertIn(mode, calls[0])
            schema = calls[0][mode] if mode == 'guided_json' else calls[0][mode]['json']
            self.assertEqual(schema, answer_schema(row()))
            self.assertEqual(result[0]['usage']['total_tokens'], 21)
            self.assertIn('including', result[0]['text'])

    def test_numeric_and_invalid(self):
        b = self.backend()
        for task, answer, expected in [('regression', -12.5, -12.5), ('probability', 0.7, 0.7),
                                       ('probability', 1.1, None), ('regression', True, None),
                                       ('regression', float('nan'), None), ('classification', 'C', None)]:
            with self.subTest(task=task, answer=answer), patch.object(b, '_request', return_value=completion(answer)):
                out = b.generate([row(task_type=task)], 'Solve', [1])[0]
                self.assertEqual(out['prediction'], expected)
                self.assertEqual(out['valid'], expected is not None)

    def test_strict_json_and_truncated_output(self):
        b = self.backend()
        for text in ['{"reasoning":"x","answer":"A","answer":"B"}',
                     '{"reasoning":"x","answer":"A","gold":"A"}', 'not json']:
            response = completion()
            response['choices'][0]['message']['content'] = text
            with patch.object(b, '_request', return_value=response):
                out = b.generate([row()], 'Solve', [1])[0]
            self.assertIsNone(out['prediction'])
            self.assertEqual(out['text'], text)
        with patch.object(b, '_request', return_value=completion(finish='length')):
            self.assertFalse(b.generate([row()], 'Solve', [1])[0]['valid'])

    def test_no_retry_and_strict_mode(self):
        for allow in (True, False):
            b = self.backend(allow_invalid=allow)
            with patch.object(b, '_request', side_effect=URLError('offline')) as req:
                if allow:
                    out = b.generate([row()], 'Solve', [1])[0]
                    self.assertIn('offline', out['error'])
                    self.assertEqual(out['request_attempts'], 1)
                else:
                    with self.assertRaises(URLError):
                        b.generate([row()], 'Solve', [1])
                self.assertEqual(req.call_count, 1)

    def test_question_only_truncation_and_no_gold(self):
        b = self.backend(context_length=1600, max_tokens=100, judge_max_tokens=100)
        candidate = {'prediction': 'A', 'text': 'FULL CANDIDATE ' * 30, 'usage': {'total_tokens': 30}}
        r = row(candidates=[candidate], answer='GOLD_POISON', reference_feedback='FEEDBACK_POISON')
        r['question'] = 'HEAD' + 'x' * 3000 + 'TAIL'
        with patch.object(b, '_request', return_value=completion()), self.assertLogs('legal_prompt_optimization.simple_backend'):
            out = b.generate([r], 'Judge', [1], judge=True)[0]
        self.assertTrue(out['valid'])
        content = out['messages'][1]['content']
        self.assertIn('HEAD', content)
        self.assertIn('TAIL', content)
        self.assertIn(candidate['text'], content)
        self.assertNotIn('POISON', content)
        self.assertGreater(out['truncation']['question_chars_removed'], 0)
        self.assertLessEqual(out['truncation']['prompt_tokens_after'], 1500)

    def test_untrimmable_candidates(self):
        b = self.backend(context_length=500, max_tokens=100, judge_max_tokens=100)
        with patch.object(b, '_request') as req:
            out = b.generate([row(candidates=[{'text': 'x' * 2000}])], 'Judge', [1], judge=True)[0]
        self.assertIn('ContextLimitError', out['error'])
        req.assert_not_called()

    def test_reflect_separate_endpoint_and_no_clipping(self):
        b = self.backend()
        for prompt in ('full feedback', [{'role': 'system', 'content': 'Reflect'}, {'role': 'user', 'content': 'full feedback'}]):
            with patch.object(b, '_request', return_value=completion()) as req:
                self.assertIsInstance(b.reflect(prompt), str)
                args, kwargs = req.call_args
                self.assertEqual(args[0], 'http://proposer')
                self.assertEqual(args[2]['model'], 'gemma')
                self.assertEqual(args[2]['messages'], [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt)
                self.assertTrue(kwargs['proposer'])
        b = self.backend(proposer_context_length=100, proposer_max_tokens=20)
        with patch.object(b, '_request') as req:
            with self.assertRaisesRegex(ContextLimitError, 'no clipping or retry'):
                b.reflect('x' * 81)
            req.assert_not_called()

    def test_timeout_and_endpoint_credentials(self):
        b = self.backend(timeout=7, api_key='student-key', proposer_api_key='proposer-key')
        with patch('legal_prompt_optimization.simple_backend.urlopen', side_effect=TimeoutError('timed out')) as op:
            out = b.generate([row()], 'Solve', [1])[0]
            self.assertIn('TimeoutError', out['error'])
            self.assertEqual(op.call_count, 1)
            request = op.call_args.args[0]
            self.assertEqual(request.get_header('Authorization'), 'Bearer student-key')
            self.assertEqual(op.call_args.kwargs['timeout'], 7)
        with patch('legal_prompt_optimization.simple_backend.urlopen', side_effect=TimeoutError('timed out')) as op:
            with self.assertRaises(TimeoutError):
                b.reflect('Full feedback')
            self.assertEqual(op.call_count, 1)
            self.assertEqual(op.call_args.args[0].get_header('Authorization'), 'Bearer proposer-key')

    def test_proposer_incomplete_is_not_accepted(self):
        b = self.backend()
        with patch.object(b, '_request', return_value=completion(finish='length')) as req:
            with self.assertRaisesRegex(RuntimeError, 'Incomplete proposer'):
                b.reflect('feedback')
            self.assertEqual(req.call_count, 1)

    def test_input_lengths(self):
        with self.assertRaises(ValueError):
            self.backend().generate([row()], 'Solve', [])
        self.assertEqual(self.backend().generate([], 'Solve', []), [])

    def test_real_urllib_preflight_and_tokenization(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                calls.append((self.path, None))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': 'phi'}]} if self.path == '/v1/models' else {}).encode())
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path, payload))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'count': 50} if self.path == '/tokenize' else completion()).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            b = SimpleBackend(f'http://127.0.0.1:{server.server_port}/v1', student_model='phi')
            self.assertTrue(b.preflight(include_proposer=False)['student']['healthy'])
            self.assertNotIn('/v1/chat/completions', [p for p, _ in calls])
            self.assertEqual(b.generate([row()], 'Solve', [42])[0]['prediction'], 'A')
            self.assertEqual([p for p, _ in calls].count('/v1/chat/completions'), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
