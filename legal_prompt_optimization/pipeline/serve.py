"""Foreground vLLM launcher with explicit API and environment checks."""

import json
import os
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from .config import load


def environment(python):
    """Inspect the serving interpreter without importing GPU libraries."""
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                (
                    "import importlib.metadata as m,json; "
                    "print(json.dumps({k:m.version(k) for k in "
                    "['vllm','torch','transformers','xgrammar']}))"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RuntimeError(
            "Cannot inspect the vLLM environment. Install the project with "
            "pip install -e '.[data,vllm]' in the active environment; "
            "--python can override the interpreter. See docs/serving.md."
        ) from error


def compatible(config, role, versions):
    version = tuple(int(v) for v in versions["vllm"].split("+")[0].split(".")[:2])
    if role == "student":
        mode = config["generation"]["schema_mode"]
        # vLLM removed guided_* request fields in 0.12.
        uses_guided = mode == "guided_json" or (
            config["generation"]["answer_forcing"]["enabled"]
            and mode != "structured_outputs"
        )
        if uses_guided and version >= (0, 12):
            raise ValueError(
                "This config sends obsolete guided_* fields. Use an appropriate server "
                "or select structured_outputs for vLLM 0.33.0 in a new run."
            )
        if mode == "structured_outputs" and version < (0, 12):
            raise ValueError("structured_outputs requires the modern vLLM environment")
    if (
        role == "proposer"
        and "gemma-4" in config["models"][role]["model"].lower()
        and version < (0, 33)
    ):
        raise ValueError("Use the pinned vLLM 0.33.0 environment for Gemma-4")


def command(args):
    c = load(args.config)
    m = c["models"][args.role]
    url = urlparse(m["endpoint"])
    if (
        url.scheme != "http"
        or url.path.rstrip("/") not in ["", "/v1"]
        or url.query
        or url.fragment
        or url.username
    ):
        raise ValueError(
            "Launch helper requires an http://localhost:PORT endpoint, optionally ending /v1"
        )
    if m["api_key_env"]:
        raise ValueError(
            "For authenticated servers, launch vLLM directly with your authentication settings; the helper launches loopback-only servers"
        )
    if url.hostname not in ["localhost", "127.0.0.1", "::1"]:
        raise ValueError(
            "Launch helper binds localhost only; configure a loopback endpoint"
        )
    if args.gpu < 0 or not 0 < args.memory_fraction < 1 or args.max_sequences < 1:
        raise ValueError("Invalid GPU/memory/concurrency setting")
    cmd = [
        args.python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        url.hostname,
        "--port",
        str(url.port or 8000),
        "--model",
        m["model"],
        "--served-model-name",
        m["model"],
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(m["context_tokens"]),
        "--gpu-memory-utilization",
        str(args.memory_fraction),
        "--max-num-seqs",
        str(args.max_sequences),
    ]
    if m["revision"]:
        cmd += ["--revision", m["revision"]]
    if args.role == "student":
        mode = c["generation"]["schema_mode"]
        if mode == "guided_json":
            cmd += ["--guided-decoding-backend", "xgrammar"]
        elif mode == "structured_outputs":
            cmd += ["--structured-outputs-config", '{"backend":"xgrammar"}']
    elif "gemma-4" in m["model"].lower():
        cmd += ["--language-model-only"]
    if args.role == "proposer" and c["generation"]["proposer"]["thinking"]:
        if "gemma" not in m["model"].lower():
            raise ValueError(
                "Thinking launch helper currently supports Gemma; start other proposer servers manually"
            )
        cmd += [
            "--reasoning-parser",
            "gemma4",
            "--default-chat-template-kwargs",
            '{"enable_thinking":true}',
        ]
    env = {"CUDA_VISIBLE_DEVICES": str(args.gpu)}
    if args.phi_eos_alignment:
        if "Phi-4-mini-instruct" not in m["model"]:
            raise ValueError("Phi EOS workaround requested for another model")
        runtime = Path(__file__).resolve().parents[1] / "runtime"
        env.update(
            LEGAL_PHI_EOS_ALIGNMENT="1",
            PYTHONPATH=str(runtime) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
    return cmd, env


def serve(args):
    cmd, env = command(args)
    if args.dry_run:
        print(
            " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            + " "
            + shlex.join(cmd)
        )
        return cmd
    versions = environment(args.python)
    compatible(load(args.config), args.role, versions)
    print("Serving environment: " + json.dumps(versions), flush=True)
    return subprocess.run(cmd, env={**os.environ, **env}, check=True).returncode
