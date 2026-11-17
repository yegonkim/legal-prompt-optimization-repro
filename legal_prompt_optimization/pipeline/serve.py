"""Optional foreground vLLM launcher; install serving dependencies separately."""

import os
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from .config import load


def command(args):
    c = load(args.config)
    m = c["models"][args.role]
    url = urlparse(m["endpoint"])
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
    return subprocess.run(cmd, env={**os.environ, **env}, check=True).returncode
