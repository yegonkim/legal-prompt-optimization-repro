"""Check the Git index for local artifacts and obvious credential material.

Checks the proposed tracked tree, not ignored files or historical commits.
Never prints matched credential values. This is not a full secret-audit guarantee.
"""

import re
import subprocess
from pathlib import Path


def main():
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    paths = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    errors = []
    secrets = [
        re.compile(p)
        for p in [
            r"gh[pousr]_[A-Za-z0-9]{30,}",
            r"github_pat_[A-Za-z0-9_]{40,}",
            r"hf_[A-Za-z0-9]{30,}",
            r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        ]
    ]
    for name in filter(None, paths):
        p = Path(name)
        if "manifests" in p.parts:
            errors.append(
                (
                    name,
                    "saved dataset membership lists are not part of the public pipeline",
                )
            )
        if p.parts[0] in [
            "runs",
            "data",
            "downloads",
            "msv_legal",
            ".source-cache",
            ".hf_home",
        ] or name in ["MACHINE_HANDOFF.md", "ADVANCED.md"]:
            errors.append((name, "local-only path tracked"))
        if p.suffix in [".pt", ".safetensors", ".bin", ".log", ".gz", ".whl", ".pyc"]:
            errors.append((name, "binary/log/archive artifact tracked"))
        if p.suffix == ".jsonl" and p.parts[0] != "examples":
            errors.append((name, "raw JSONL tracked"))
        if p.parts[0] == "results" and (len(p.parts) < 2 or p.parts[1] != "sequential"):
            errors.append((name, "uncurated result artifact tracked"))
        body = subprocess.check_output(["git", "show", ":" + name], cwd=root)
        if len(body) > 2_000_000:
            errors.append((name, "file exceeds2MB public-tree limit"))
        text = body.decode("utf-8", errors="replace")
        if any(pattern.search(text) for pattern in secrets):
            errors.append((name, "possible credential material"))
    for name, reason in errors:
        print(f"{name}: {reason}")
    if errors:
        raise SystemExit(1)
    print(
        f"Public tree checks passed for {len(list(filter(None, paths)))} tracked files (current index only)."
    )


if __name__ == "__main__":
    main()
