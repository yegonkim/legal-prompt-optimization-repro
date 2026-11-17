import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from legal_prompt_optimization.pipeline.backend import Backend
from legal_prompt_optimization.pipeline.config import DEFAULTS


class Endpoint(unittest.TestCase):
    def test_forcing_is_sampled_cached_and_optional_without_resampling(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    response = {"count": 100}
                else:
                    calls.append(body)
                    forcing = body.get("continue_final_message", False)
                    response = {
                        "id": "request" + str(len(calls)),
                        "usage": {"completion_tokens": 3},
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop" if forcing else "length",
                                "message": {
                                    "content": '"B"}'
                                    if forcing
                                    else '{"reasoning":"partial'
                                },
                            }
                        ],
                    }
                content = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                c = copy.deepcopy(DEFAULTS)
                c["models"]["student"]["endpoint"] = (
                    f"http://127.0.0.1:{server.server_port}"
                )
                b = Backend(c, tmp)
                row = {
                    "record_id": "x",
                    "question": "q",
                    "task_type": "classification",
                    "public_labels": ["A", "B"],
                }
                result = b.generate([row], "solve", [42])[0]
                self.assertTrue(result["valid"])
                self.assertEqual(result["prediction"], "B")
                self.assertTrue(result["answer_forcing"]["recovered"])
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[1]["temperature"], 0.7)
                self.assertEqual(calls[1]["top_p"], 0.95)
                self.assertFalse(calls[1]["add_generation_prompt"])
                self.assertEqual(calls[1]["guided_choice"], ['"A"}', '"B"}'])
                stored = json.loads(Path(result["cache_record"]).read_text())
                self.assertEqual(stored["original_output"]["finish_reason"], "length")
                self.assertEqual(result, b.generate([row], "solve", [42])[0])
                self.assertEqual(len(calls), 2)
                c["generation"]["answer_forcing"]["enabled"] = False
                original = Backend(c, tmp).generate([row], "solve", [42])[0]
                self.assertFalse(original["valid"])
                self.assertEqual(len(calls), 2)  # Same original request reused.
                c["generation"]["judge"]["temperature"] = 0.4
                c["generation"]["judge"]["max_tokens"] = 256
                judge_row = {**row, "candidates": [result] * 4}
                Backend(c, tmp).generate([judge_row], "judge", [43], judge=True)
                self.assertEqual(calls[-1]["temperature"], 0.4)
                self.assertEqual(calls[-1]["max_tokens"], 256)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
