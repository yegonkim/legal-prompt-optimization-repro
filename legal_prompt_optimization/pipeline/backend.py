"""Configurable cached endpoint backend. No research scripts, host paths or env overrides."""

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from ..simple_backend import ContextLimitError, SimpleBackend
from .config import inference_signature
from .io import digest, dumps, read, write


@lru_cache(maxsize=32)
def compact_grammar(schema_json):
    try:
        import xgrammar
    except ImportError as error:
        raise RuntimeError(
            "Reference compact grammar requires the optional grammar extra"
        ) from error
    return str(
        xgrammar.Grammar.from_json_schema(
            schema_json, any_whitespace=False, indent=None, separators=(",", ":")
        )
    )


def reasoning_prefix(text):
    match = re.search(r'"reasoning"\s*:\s*', text)
    if not match:
        return ""
    value = text[match.end() :]
    try:
        answer, _ = json.JSONDecoder().raw_decode(value)
        return answer if isinstance(answer, str) else text
    except ValueError:
        for suffix in ['"', '\\"']:
            try:
                answer = json.loads(value + suffix)
                if isinstance(answer, str):
                    return answer
            except ValueError:
                pass
        return text


class Backend(SimpleBackend):
    def __init__(self, config, cache):
        self.config = config
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.local = threading.local()
        self.count_lock = threading.Lock()
        self.counts = {
            "response_requests": 0,
            "judge_requests": 0,
            "forcing_requests": 0,
            "proposer_requests": 0,
            "response_cache_hits": 0,
            "record_cache_hits": 0,
        }
        m = config["models"]
        g = config["generation"]
        r = config["runtime"]

        def tokenizer(role):
            spec = m[role]
            if spec["tokenizer"] is None:
                return None
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                spec["tokenizer"],
                revision=spec["revision"],
                local_files_only=r["offline"],
            )

        def key(role):
            name = m[role]["api_key_env"]
            if name and not os.environ.get(name):
                raise ValueError(f"Missing API key environment variable {name}")
            return os.environ.get(name) if name else None

        super().__init__(
            m["student"]["endpoint"],
            m["proposer"]["endpoint"],
            student_model=m["student"]["model"],
            proposer_model=m["proposer"]["model"],
            student_tokenizer=tokenizer("student"),
            proposer_tokenizer=tokenizer("proposer"),
            context_length=m["student"]["context_tokens"],
            proposer_context_length=m["proposer"]["context_tokens"],
            max_tokens=g["response"]["max_tokens"],
            judge_max_tokens=g["judge"]["max_tokens"],
            proposer_max_tokens=g["proposer"]["max_tokens"],
            temperature=g["response"]["temperature"],
            top_p=g["response"]["top_p"],
            proposer_temperature=g["proposer"]["temperature"],
            concurrency=r["concurrency"],
            timeout=r["timeout_seconds"],
            api_key=key("student"),
            proposer_api_key=key("proposer"),
            max_physical_calls=r["max_physical_calls"],
            max_seconds=r["max_seconds"],
            allow_invalid=True,
            bounded_numeric_output=False,
        )

    def _request(self, endpoint, path, payload=None, *, proposer=False):
        if path != "/v1/chat/completions":
            return super()._request(endpoint, path, payload, proposer=proposer)
        role = "proposer" if proposer else getattr(self.local, "role", "response")
        settings = self.config["generation"][
            "answer_forcing" if role == "forcing" else role
        ]
        payload = {**payload}
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ]:
            if key in settings and settings[key] is not None:
                payload[key] = settings[key]
        if proposer:
            payload["chat_template_kwargs"] = {"enable_thinking": settings["thinking"]}
        mode = self.config["generation"]["schema_mode"]
        if (
            not proposer
            and self.config["protocol"] == "legal_gepa_v1"
            and "guided_json" in payload
        ):
            grammar = compact_grammar(json.dumps(payload.pop("guided_json")))
            if mode == "structured_outputs":
                payload["structured_outputs"] = {"grammar": grammar}
            else:
                payload["guided_grammar"] = grammar

        if not proposer and "guided_json" in payload:
            schema = payload.pop("guided_json")
            if mode == "guided_json":
                payload["guided_json"] = schema
            elif mode == "structured_outputs":
                payload["structured_outputs"] = {"json": schema}
            elif mode == "response_format":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "strict": True, "schema": schema},
                }
        if role == "forcing" and mode == "structured_outputs":
            if "guided_choice" in payload:
                payload["structured_outputs"] = {"choice": payload.pop("guided_choice")}
            elif "guided_regex" in payload:
                payload["structured_outputs"] = {"regex": payload.pop("guided_regex")}
        identity = {
            "role": role,
            "model": self.config["models"]["proposer" if proposer else "student"],
            "payload": payload,
        }
        identity["model"] = {
            k: v
            for k, v in identity["model"].items()
            if k not in ["endpoint", "api_key_env", "tokenizer"]
        }
        dest = self.cache / "responses" / (digest(identity) + ".json")
        if dest.exists():
            with self.count_lock:
                self.counts["response_cache_hits"] += 1
            return read(dest)
        response = super()._request(endpoint, path, payload, proposer=proposer)
        write(
            dest, response
        )  # Raw outputs retained even if parsing subsequently fails.
        with self.count_lock:
            self.counts[role + "_requests"] += 1
            write(self.cache / "counts.json", self.counts)
        return response

    def _prepare(self, row, instruction, judge, cap):
        if self.config["protocol"] == "legal_gepa_v1":
            from ..core import OUTPUT_INSTRUCTION, schema

            messages = [
                {
                    "role": "system",
                    "content": instruction + "\n\n" + OUTPUT_INSTRUCTION,
                },
                {"role": "user", "content": row["question"]},
            ]
            count = self._count(messages)
            if count + cap > self.context_length:
                raise ContextLimitError("Reference input exceeds context; no clipping")
            return (
                messages,
                schema(row["public_labels"]),
                {
                    "truncated": False,
                    "prompt_tokens_before": count,
                    "prompt_tokens_after": count,
                },
            )
        messages, schema, info = super()._prepare(row, instruction, judge, cap)
        if info["truncated"]:
            raise ContextLimitError("Input exceeds context; no truncation is permitted")
        return messages, schema, info

    def force(self, row, instruction, seed, record, judge):
        g = self.config["generation"]["answer_forcing"]
        if not g["enabled"] or record.get("finish_reason") != "length":
            return record
        result = {
            **record,
            "original_output": record,
            "answer_forcing": {"attempted": True, "recovered": False},
        }
        try:
            messages, schema, _ = self._prepare(
                row,
                instruction,
                judge,
                self.judge_max_tokens if judge else self.max_tokens,
            )
            prefix = (
                '{"reasoning": '
                + dumps(reasoning_prefix(record["text"]))
                + ', "answer": '
            )
            messages.append({"role": "assistant", "content": prefix})
            if self.student_tokenizer is not None:
                count = len(
                    self.student_tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        continue_final_message=True,
                    )
                )
            else:
                count = self._request(
                    self.student_endpoint,
                    "/tokenize",
                    {
                        "model": self.student_model,
                        "messages": messages,
                        "add_generation_prompt": False,
                        "continue_final_message": True,
                    },
                )["count"]
            if count + g["max_tokens"] > self.context_length:
                raise ContextLimitError("No room for forced answer continuation")
            payload = {
                "model": self.student_model,
                "messages": messages,
                "n": 1,
                "seed": seed,
                "max_tokens": g["max_tokens"],
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
            rule = schema["properties"]["answer"]
            if "enum" in rule:
                payload["guided_choice"] = [json.dumps(v) + "}" for v in rule["enum"]]
            else:
                payload["guided_regex"] = (
                    r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?\}"
                )
            self.local.role = "forcing"
            response = self._request(
                self.student_endpoint, "/v1/chat/completions", payload
            )
            choice = response["choices"][0]
            text = prefix + (choice["message"].get("content") or "")
            result["answer_forcing"].update(
                usage=response.get("usage"), finish_reason=choice.get("finish_reason")
            )
            if choice["finish_reason"] != "stop":
                raise ValueError("Forced continuation hit its cap")
            prediction, reasoning = self._parse(text, schema)
            result.update(
                prediction=prediction,
                reasoning=reasoning,
                text=text,
                valid=True,
                error=None,
                finish_reason="stop",
            )
            result["answer_forcing"]["recovered"] = True
        except ContextLimitError:
            raise
        except ValueError as error:
            result["answer_forcing"]["error"] = repr(error)
        return result

    def generate(self, rows, instructions, seeds, judge=False):
        rows = list(rows)
        seeds = list(seeds)
        instructions = (
            [instructions] * len(rows)
            if isinstance(instructions, str)
            else list(instructions)
        )
        if not len(rows) == len(seeds) == len(instructions):
            raise ValueError("Mismatched request lengths")
        base_generate = super().generate

        def one(args):
            row, instruction, seed = args
            key = digest(
                {
                    "recipe": inference_signature(self.config),
                    "row": row,
                    "instruction": instruction,
                    "seed": seed,
                    "judge": judge,
                }
            )
            dest = self.cache / "records" / (key + ".json")
            if dest.exists():
                with self.count_lock:
                    self.counts["record_cache_hits"] += 1
                record = read(dest)
            else:
                self.local.role = "judge" if judge else "response"
                record = base_generate([row], [instruction], [seed], judge=judge)[0]
                # Infrastructure failures must stop the run, not become model scores.
                if record.get("finish_reason") is None:
                    raise RuntimeError(
                        "Generation request failed: " + str(record.get("error"))
                    )
                record.pop("messages", None)
                record = self.force(row, instruction, seed, record, judge)
                write(dest, record)
            result = {k: v for k, v in record.items() if k != "original_output"}
            result["cache_record"] = str(dest)
            return result

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(one, zip(rows, instructions, seeds)))

    def reflect(self, prompt, *, call_number=None):
        forcing = self.config["generation"]["answer_forcing"]
        if forcing["enabled"]:
            prompt += "\n\nGeneration policy: length-truncated responses receive one sampled answer-prefill continuation. Scores and feedback use the final prediction after this continuation. Preserve the JSON reasoning/answer contract."
        messages = [{"role": "user", "content": prompt}]
        if (
            self._count(messages, proposer=True) + self.proposer_max_tokens
            > self.proposer_context_length
        ):
            raise ContextLimitError(
                "Reflection feedback exceeds proposer context; no truncation"
            )
        seed = self.config["generation"]["proposer"]["seed_offset"]
        if seed is not None and call_number is None:
            raise ValueError("Seeded reflection requires an explicit call number")
        response = self._request(
            self.proposer_endpoint,
            "/v1/chat/completions",
            {
                "model": self.proposer_model,
                "messages": messages,
                "max_tokens": self.proposer_max_tokens,
                **({"seed": seed + call_number} if seed is not None else {}),
            },
            proposer=True,
        )
        c = response["choices"][0]
        text = c["message"].get("content")
        if (
            c["finish_reason"] != "stop"
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Incomplete or empty proposer output")
        return text
