"""Prepare ecthr_npd from pinned public sources."""

import sys

from legal_prompt_optimization.pipeline.cli import main

if __name__ == "__main__":
    main(["prepare", "--dataset", "ecthr_npd", *sys.argv[1:]])
