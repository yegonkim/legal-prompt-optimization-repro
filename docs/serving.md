# Model endpoints and optional vLLM launch helpers

The optimization runner is separate from model serving. Keep existing servers if you have them; configure `models.student.endpoint` and `models.proposer.endpoint` in YAML. A server must expose OpenAI-compatible chat completions and the configured structured-output mode. If local tokenizers are not configured, `/tokenize` is also required. Answer forcing additionally requires `continue_final_message`, `add_generation_prompt: false`, and guided choice/regex continuation.

The default models are Phi-4-mini-instruct for responses/judges and Gemma-4-12B-it for reflection. Both model revisions are pinned in the example configs. Model access approval and Hugging Face authentication, where required, are handled in the serving environment.

The endpoint contract is `/health`, `/v1/models`, and `/v1/chat/completions`, plus `/tokenize` when no local tokenizer is configured. The backend sends vLLM decoding extensions such as `top_k` and `repetition_penalty`. A generic hosted API may require an adapter even with answer forcing disabled; this is not a universal provider SDK.

## Optional foreground launcher

Install vLLM in a separate environment, following the requirements for your GPU and model. Gemma requires a newer vLLM with its architecture and thinking parser. The two servers may use different environments.

Preview the commands first:

```bash
legal-prompts serve --config configs/pipeline/ledgar.yaml --role student \
  --python /path/to/student-env/bin/python --gpu 0 \
  --memory-fraction 0.40 --max-sequences 8 --dry-run

legal-prompts serve --config configs/pipeline/ledgar.yaml --role proposer \
  --python /path/to/proposer-env/bin/python --gpu 0 \
  --memory-fraction 0.55 --max-sequences 1 --dry-run
```

Remove `--dry-run` to launch each foreground process, in separate terminals or your service manager. The helpers bind only to the loopback host/port in the config. They do not inspect/stop existing services, install dependencies, manage experiment scheduling, or guarantee that the chosen models/context lengths fit your GPU. CUDA graphs remain enabled by vLLM's normal defaults; eager execution is not forced.

The launcher passes model/revision, context length, BF16, GPU-memory fraction and maximum sequences explicitly. Gemma thinking adds the Gemma4 reasoning parser and the thinking chat-template flag. For another thinking model, launch the server manually with its appropriate parser. Additional server-specific flags can likewise be supplied by launching vLLM directly from the printed command.

`--phi-eos-alignment` explicitly enables that existing hook for Phi-4-mini-instruct; it is opt-in and should not be assumed necessary on newer servers. Server upgrades can change numerical results.

## Endpoint-only use

Select `generation.schema_mode` to match your server:

- `guided_json`: older vLLM API.
- `structured_outputs`: newer vLLM API.
- `response_format`: JSON-schema response format.
- `none`: request JSON in the prompt and validate it afterward.

Answer forcing is a vLLM-style extension; generic hosted APIs may not support assistant prefill. Disable it for endpoints without that capability. A preflight verifies health/model identifiers/tokenization, while a small real run is needed to verify generation features.

Credentials are configured as environment-variable names, for example `api_key_env: MY_MODEL_API_KEY`. The value is read at runtime and never saved into the resolved config or cache identity. Do not place secrets in model URLs or YAML values.
