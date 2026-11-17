"""Prepare sara_entailment from pinned public sources."""

import sys

from legal_prompt_optimization.pipeline.cli import main

if __name__ == "__main__":
    main(["prepare", "--dataset", "sara_entailment", *sys.argv[1:]])
