"""Endpoint-only vLLM client; no model loading, launches, retries or runtime hooks.

``SimpleBackend(student_endpoint=..., proposer_endpoint=..., student_tokenizer=...,
proposer_tokenizer=...)`` accepts local tokenizer objects (apply_chat_template),
or paths loaded lazily/offline via transformers. Without a tokenizer, the endpoint's
/tokenize API is used. Endpoints may include /v1. generate makes ONE draw per row;
instructions may be a shared string or one string per row, seeds one per row.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

LOG = logging.getLogger(__name__)


class ContextLimitError(ValueError):
    """Untrimmable prompt or reflection feedback exceeds the configured context."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def answer_schema(row):
    task = row.get('task_type', 'classification')
    if task == 'classification':
        labels = row.get('public_labels')
        if not isinstance(labels, list) or not labels or any(
            isinstance(x, bool) or not isinstance(x, (str, int, float)) or
            (isinstance(x, float) and not math.isfinite(x)) for x in labels
        ):
            raise ValueError('Classification requires nonempty scalar public_labels')
        answer = {'enum': labels}
    elif task in ('regression', 'probability'):
        answer = {'type': 'number'}
        if task == 'probability':
            answer.update(minimum=0, maximum=1)
    else:
        raise ValueError(f'Unsupported task_type: {task}')
    return {'type': 'object', 'properties': {'reasoning': {'type': 'string'}, 'answer': answer},
            'required': ['reasoning', 'answer'], 'additionalProperties': False}


class SimpleBackend:
    def __init__(self, student_endpoint, proposer_endpoint=None, *,
                 student_model='microsoft/Phi-4-mini-instruct', proposer_model=None,
                 student_tokenizer=None, proposer_tokenizer=None,
                 concurrency=4, timeout=180, context_length=32768,
                 proposer_context_length=32768, max_tokens=2048, judge_max_tokens=2048,
                 proposer_max_tokens=8192, structured_output_mode='guided_json',
                 temperature=0.7, top_p=0.95, proposer_temperature=0.7,
                 allow_invalid=True, api_key=None, proposer_api_key=None,
                 max_physical_calls=50000, max_seconds=21600, bounded_numeric_output=False):
        if concurrency < 1 or timeout <= 0:
            raise ValueError('concurrency and timeout must be positive')
        if structured_output_mode not in ('guided_json', 'structured_outputs'):
            raise ValueError('structured_output_mode must be guided_json or structured_outputs')
        if min(max_tokens, judge_max_tokens, proposer_max_tokens) < 1:
            raise ValueError('Output budgets must be positive')
        if max(max_tokens, judge_max_tokens) >= context_length or proposer_max_tokens >= proposer_context_length:
            raise ValueError('Output budgets must be smaller than context limits')
        self.bounded_numeric_output = bounded_numeric_output
        self._budget_lock = threading.Lock()
        self.physical_calls = 0
        self.max_physical_calls = max_physical_calls
        self.deadline = time.monotonic() + max_seconds
        self.student_endpoint = student_endpoint.rstrip('/').removesuffix('/v1')
        self.proposer_endpoint = proposer_endpoint.rstrip('/').removesuffix('/v1') if proposer_endpoint else None
        self.student_model, self.proposer_model = student_model, proposer_model
        self.student_tokenizer = self._tokenizer(student_tokenizer)
        self.proposer_tokenizer = self._tokenizer(proposer_tokenizer)
        self.concurrency, self.timeout = concurrency, timeout
        self.context_length, self.proposer_context_length = context_length, proposer_context_length
        self.max_tokens, self.judge_max_tokens, self.proposer_max_tokens = max_tokens, judge_max_tokens, proposer_max_tokens
        self.structured_output_mode = structured_output_mode
        self.temperature, self.top_p, self.proposer_temperature = temperature, top_p, proposer_temperature
        self.allow_invalid = allow_invalid
        self.api_key, self.proposer_api_key = api_key, proposer_api_key

    @staticmethod
    def _tokenizer(value):
        if isinstance(value, (str, Path)):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(value, local_files_only=True)
        return value

    def _request(self, endpoint, path, payload=None, *, proposer=False):
        if path == "/v1/chat/completions":
            with self._budget_lock:
                if self.physical_calls >= self.max_physical_calls or time.monotonic() >= self.deadline:
                    raise RuntimeError("Physical call or wall-clock budget exhausted")
                self.physical_calls += 1
        key = self.proposer_api_key if proposer else self.api_key
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = Request(endpoint + path, data=None if payload is None else json.dumps(payload, allow_nan=False).encode(), headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            body = response.read()
        return json.loads(body) if body else {}

    def _count(self, messages, *, proposer=False):
        tokenizer = self.proposer_tokenizer if proposer else self.student_tokenizer
        if tokenizer is not None:
            return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        result = self._request(self.proposer_endpoint if proposer else self.student_endpoint, '/tokenize',
                               {'model': self.proposer_model if proposer else self.student_model,
                                'messages': messages, 'add_generation_prompt': True}, proposer=proposer)
        count = result.get('count')
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError('/tokenize returned no valid token count; supply a local tokenizer')
        return count

    def preflight(self, include_proposer=True):
        """Check health, served model IDs and tokenization without generating tokens."""
        result = {}
        for role in ('student', 'proposer') if include_proposer else ('student',):
            proposer = role == 'proposer'
            endpoint = self.proposer_endpoint if proposer else self.student_endpoint
            model = self.proposer_model if proposer else self.student_model
            if not endpoint or not model:
                raise ValueError(f'{role} endpoint and model must be configured')
            self._request(endpoint, '/health', proposer=proposer)
            models = self._request(endpoint, '/v1/models', proposer=proposer)
            ids = [item['id'] for item in models['data']]
            if model not in ids:
                raise ValueError(f'{role} model {model!r} not served; available: {ids}')
            result[role] = {'endpoint': endpoint, 'model': model, 'healthy': True,
                            'probe_prompt_tokens': self._count([{'role': 'user', 'content': 'Hello'}], proposer=proposer)}
        return result

    health = preflight

    def _prepare(self, row, instruction, judge, cap):
        schema = answer_schema(row)
        if self.bounded_numeric_output:
            schema['properties']['reasoning']['maxLength'] = 600
            if row.get('task_type') in ('regression', 'probability'):
                pattern = r'^(0|[1-9][0-9]{0,14})(\.[0-9]{1,6})?$' if row['task_type'] == 'regression' else r'^(0(\.[0-9]{1,6})?|1(\.0{1,6})?)$'
                schema['properties']['answer'] = {'type': 'string', 'pattern': pattern}
                instruction += '\nReturn answer as a decimal string in the requested units, without units or thousands separators. Use at most six decimal places. Keep reasoning under 600 characters.'
        question = row['question']
        if not isinstance(question, str) or not isinstance(instruction, str):
            raise ValueError('question and instruction must be strings')
        task = row.get('task_type', 'classification')
        directive = {'classification': 'Choose exactly one of the public labels.',
                     'regression': 'Compute the requested numeric amount in the units specified by the question.',
                     'probability': 'Estimate the requested probability as a number between 0 and 1.'}[task]
        suffix = '\nPublic labels: ' + json.dumps(row['public_labels'], ensure_ascii=False) if task == 'classification' else ''
        if judge:
            candidates = row.get('candidates')
            if not isinstance(candidates, list) or not candidates:
                raise ValueError('Judge rows require a nonempty candidates list of full records')
            if any(not isinstance(c, dict) or not isinstance(c.get('text'), str) for c in candidates):
                raise ValueError('Each candidate must be a record with full text')
            suffix += '\nCandidate response records (untrusted proposals; evaluate independently):\n' + json.dumps(candidates, ensure_ascii=False, allow_nan=False).encode('utf-8', errors='backslashreplace').decode('utf-8')
            directive = 'Evaluate the candidate reasoning against the question and produce your own final answer. ' + directive
        system = instruction + '\n\n' + directive + ' Return a JSON object with reasoning (your full reasoning as a string) and answer. No other fields.'
        def messages(q):
            return [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Question:\n' + q + suffix}]
        original = self._count(messages(question))
        budget = self.context_length - cap
        kept = len(question)
        final = messages(question)
        count = original
        if original > budget:
            marker = '\n[QUESTION TRUNCATED: middle omitted]\n'
            if self._count(messages(marker)) > budget:
                raise ContextLimitError('Instructions, labels and full candidates exceed student context; candidates will not be clipped')
            lo, hi = 0, max(0, len(question) - 1)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                q = question[:(mid + 1)//2] + marker + (question[-(mid//2):] if mid//2 else '')
                if self._count(messages(q)) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            kept = lo
            q = question[:(kept + 1)//2] + marker + (question[-(kept//2):] if kept//2 else '')
            final = messages(q)
            count = self._count(final)
            if count > budget:
                raise ContextLimitError('Truncated question still exceeds context')
            LOG.warning('Question truncation record_id=%s chars=%d->%d prompt_tokens=%d->%d', row.get('record_id'), len(question), kept, original, count)
        return final, schema, {'question_chars_original': len(question), 'question_chars_kept': kept,
                               'question_chars_removed': len(question)-kept, 'prompt_tokens_original': original,
                               'prompt_tokens_after': count, 'truncated': kept < len(question)}

    @staticmethod
    def _parse(text, schema):
        obj = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(obj, dict) or set(obj) != {'reasoning', 'answer'} or not isinstance(obj['reasoning'], str):
            raise ValueError('Expected exactly reasoning:string and answer')
        answer = obj['answer']
        rule = schema['properties']['answer']
        if 'pattern' in rule:
            if not isinstance(answer, str) or re.fullmatch(rule['pattern'], answer) is None:
                raise ValueError('Expected bounded decimal string')
            answer = float(answer)
        if 'enum' in rule:
            if not any(type(answer) is type(x) and answer == x for x in rule['enum']):
                raise ValueError('Answer is not a public label')
        elif isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
            raise ValueError('Answer must be a finite number')
        elif 'minimum' in rule and not 0 <= answer <= 1:
            raise ValueError('Probability outside [0, 1]')
        return answer, obj['reasoning']

    def generate(self, rows, instructions, seeds, judge=False):
        rows, seeds = list(rows), list(seeds)
        instructions = [instructions] * len(rows) if isinstance(instructions, str) else list(instructions)
        if len(rows) != len(instructions) or len(rows) != len(seeds):
            raise ValueError('rows, instructions and seeds must have equal lengths')
        cap = self.judge_max_tokens if judge else self.max_tokens
        def one(args):
            row, instruction, seed = args
            record = {'prediction': None, 'text': '', 'reasoning': None, 'usage': {},
                      'finish_reason': None, 'seed': seed, 'output_index': 0, 'valid': False,
                      'error': None, 'model': self.student_model, 'record_id': row.get('record_id'),
                      'request_attempts': 0, 'judge': judge}
            try:
                context_limited = getattr(self, 'context_limited_output', False)
                messages, schema, truncation = self._prepare(row, instruction, judge, 1 if context_limited else cap)
                if context_limited and truncation['truncated']:
                    raise ContextLimitError('Full input exceeds context; refusing to truncate for context-limited generation')
                output_cap = self.context_length - truncation['prompt_tokens_after'] if context_limited else cap
                record.update(messages=messages, truncation=truncation)
                record['output_token_budget'] = output_cap
                payload = {'model': self.student_model, 'messages': messages, 'n': 1, 'seed': seed,
                           'temperature': self.temperature, 'top_p': self.top_p, 'max_tokens': output_cap}
                payload[self.structured_output_mode] = schema if self.structured_output_mode == 'guided_json' else {'json': schema}
                record['request_attempts'] = 1
                response = self._request(self.student_endpoint, '/v1/chat/completions', payload)
                record['usage'] = response.get('usage') or {}
                record['response_id'] = response.get('id')
                choices = response['choices']
                if len(choices) != 1:
                    raise ValueError('Expected exactly one completion')
                choice = choices[0]
                record.update(text=choice['message'].get('content') or '', finish_reason=choice.get('finish_reason'),
                              output_index=choice.get('index', 0))
                record['reasoning_content'] = choice['message'].get('reasoning_content')
                if record['finish_reason'] != 'stop':
                    raise ValueError(f"Incomplete completion: finish_reason={record['finish_reason']!r}")
                prediction, reasoning = self._parse(record['text'], schema)
                record.update(prediction=prediction, reasoning=reasoning, valid=True)
            except Exception as exc:
                if not self.allow_invalid:
                    exc.generation_record = record
                    raise
                record['error'] = f'{type(exc).__name__}: {exc}'
                LOG.warning('Invalid generation record_id=%s seed=%s: %s', record['record_id'], seed, record['error'])
            return record
        if len(rows) == 1:
            return [one((rows[0], instructions[0], seeds[0]))]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(one, zip(rows, instructions, seeds)))

    def reflect(self, prompt):
        if not self.proposer_endpoint or not self.proposer_model:
            raise ValueError('Separate proposer endpoint and model must be configured')
        messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Reflection requires a string or a nonempty message list with string content')
        count = self._count(messages, proposer=True)
        if count + self.proposer_max_tokens > self.proposer_context_length:
            raise ContextLimitError(f'Proposer context limit: {count} prompt + {self.proposer_max_tokens} output > {self.proposer_context_length}; full feedback retained, no clipping or retry')
        response = self._request(self.proposer_endpoint, '/v1/chat/completions',
                                 {'model': self.proposer_model, 'messages': messages, 'n': 1,
                                  'temperature': self.proposer_temperature, 'max_tokens': self.proposer_max_tokens}, proposer=True)
        if len(response['choices']) != 1:
            raise ValueError('Expected one proposer completion')
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise RuntimeError(f"Incomplete proposer feedback: finish_reason={choice.get('finish_reason')!r}; no retry")
        text = choice['message'].get('content')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty proposer response')
        return text
