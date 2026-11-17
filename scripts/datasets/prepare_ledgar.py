"""Prepare ledgar from pinned public sources."""

import sys

from legal_prompt_optimization.pipeline.cli import main

if __name__ == "__main__":
    main(["prepare", "--dataset", "ledgar", *sys.argv[1:]])
