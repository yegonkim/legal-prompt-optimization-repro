# vLLM model serving

vLLM is the default backend. Run `legal-prompts` in a runner environment and the two model servers in a shared serving environment. The supplied experiment configs use Phi for responses/judges and Gemma for reflection. Installing the runner does not install either server.

## Install

Run these commands from the repository root on Linux with Python 3.11 and an NVIDIA GPU. Both Phi and Gemma use **vLLM 0.33.0 in one shared serving environment**. The runner stays separate. The pinned serving wheel uses CUDA 13.0; your driver and GPU must support it, and BF16 requires Ampere or newer. See the [official vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) for hardware-specific installation.

```bash
python3.11 -m venv .venv-runner
.venv-runner/bin/python -m pip install --upgrade pip
.venv-runner/bin/python -m pip install -e '.[data,grammar]'

python3.11 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install -r requirements/serve.txt
.venv-vllm/bin/python -m pip check
.venv-vllm/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); print(torch.ones(1, device="cuda") + 1)'

source .venv-runner/bin/activate
legal-prompts --help
```

| Models | vLLM | PyTorch | Transformers | XGrammar | Student request API |
|---|---|---|---|---|---|
| Phi and Gemma | 0.33.0 | 2.14.0 | 5.20.2 | 0.2.8 | `structured_outputs` |

The requirements pin the four core serving packages, not all transitive dependencies. All supplied configs use `structured_outputs`; JSON, compact grammar, and answer-forcing choice/regex constraints use this API. See [vLLM structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/).

For a gated model, obtain access on its Hugging Face model page, then authenticate:

```bash
.venv-vllm/bin/hf auth login
```

The servers download the model revisions pinned in the config on first launch. Allow disk space and download time for both checkpoints. Set `HF_HOME` before launching if you want a shared cache. For unattended authentication, supply `HF_TOKEN` through your environment or secret manager. Never put credentials in YAML.

## Launch

Choose the experiment config you intend to run and set its `models.student.endpoint` and `models.proposer.endpoint` to the desired ports. Default ports are 8000 and 8001. Launch from the activated runner environment. These examples use one GPU per model; run each command in its own terminal:

```bash
legal-prompts serve --config configs/experiments/ledgar.yaml --role student \
  --python .venv-vllm/bin/python --gpu 0 \
  --memory-fraction 0.85 --max-sequences 8

legal-prompts serve --config configs/experiments/ledgar.yaml --role proposer \
  --python .venv-vllm/bin/python --gpu 1 \
  --memory-fraction 0.85 --max-sequences 1
```

Add `--dry-run` to print the command without inspecting the environment or starting a server. Actual launch checks the selected interpreter and rejects known API/version mismatches. The helper runs in the foreground; use your service manager for persistent servers. It does not stop existing processes or alter installed packages.

Model revision, BF16, context limit, memory fraction, maximum sequences and the structured-output backend are passed explicitly. Gemma-4 runs in language-only mode; thinking enables the `gemma4` reasoning parser and chat-template flag. CUDA graphs use vLLM defaults; eager execution is not forced. Requests run concurrently according to `runtime.concurrency`, allowing vLLM to batch them. Each draw has its own seed and `n: 1`.

Phi's BF16 weights require roughly 8 GB and Gemma's roughly 24 GB before KV cache and runtime overhead. Context length and concurrency add memory use; these are not minimum VRAM guarantees. To share a sufficiently large GPU, use the same `--gpu` with memory fractions whose sum leaves headroom, for example 0.37 and 0.55. Reduce concurrency/memory reservations according to available hardware; keep experiment context limits intact unless deliberately creating a new config. A 24 GB card cannot hold both BF16 models together.

`--phi-eos-alignment` enables the bundled opt-in Phi EOS hook. It is not enabled automatically. Server upgrades and hook changes can change generation; document them when comparing results.

## Verify real generation

The runner automatically checks health, vLLM API compatibility, served model names, context capacity and tokenization before optimization/evaluation. You do not need a separate preflight command. For a newly installed server pair, run the synthetic smoke test once:

```bash
legal-prompts smoke --config configs/experiments/ledgar.yaml --out runs/smoke-ledgar
legal-prompts smoke --config configs/experiments/parcel.yaml --out runs/smoke-parcel
```

Each smoke test uses four synthetic responses, a four-response judge and one reflection request. It tests the selected protocol, including compact grammar when configured. With answer forcing enabled, it also deliberately caps initial generations at one token and tests sampled label-choice and numeric-regex continuations for both response and judge. Only this diagnostic uses the one-token initial cap; experiment caps/config files are unchanged. It writes `smoke.json`, resolved settings and request caches locally, and exits nonzero if a required path fails. Use a new output directory for another attempt. `--student-only` skips reflection if only the actor/judge server is available.

This is a compatibility test, not a score-reproduction claim or guarantee that every dataset output will parse. It does not read any dataset, select prompts or modify experiment outputs. `legal-prompts preflight --config CONFIG.yaml` remains an optional health-only diagnostic.

To test GEPA and evaluation through the public command on a tiny synthetic dataset:

```bash
legal-prompts optimize --data examples/toy.jsonl --config examples/smoke.yaml --out runs/toy
legal-prompts evaluate --data examples/toy.jsonl --config examples/smoke.yaml \
  --winner runs/toy/winner.json --split test --out runs/toy-test
```

## Troubleshooting and existing servers

- Connection refused: wait for server startup, check its logs, and verify the configured port. Downloading weights and building CUDA graphs can take time.
- Unknown `guided_json`, `guided_choice` or `guided_regex`: use a current config with `schema_mode: structured_outputs`; these fields were removed from modern vLLM.
- Unknown Gemma architecture or `gemma4` parser: use the pinned shared serving environment.
- CUDA out of memory: check other GPU processes, model weights, context length and the combined memory reservations. The launcher does not manage GPU allocation across servers.
- Missing final reflection text: check that thinking output is parsed separately and that the configured proposer output budget is sufficient. The runner rejects truncated/empty reflection output.
- Context overflow: inputs are never silently clipped. Check the server's loaded context against the config and the full prompt plus output budget.

Existing vLLM endpoints need `/health`, `/version`, `/v1/models`, `/v1/chat/completions`, and `/tokenize` unless local tokenizers are configured. Endpoint URLs may include `/v1`. Answer forcing additionally needs assistant prefill (`continue_final_message`, `add_generation_prompt: false`) and choice/regex constraints. Generic hosted APIs are not a supported drop-in replacement.

Credentials are configured as environment-variable names using `api_key_env`; values are read at runtime and never placed in saved configs or cache identities. For authenticated or externally bound servers, start vLLM with your own server authentication/network settings, then point the runner at it. The launch helper supports unauthenticated loopback servers only and rejects authenticated configs rather than accidentally starting an unsecured server.
