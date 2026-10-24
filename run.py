"""Run the curated experiment from this checkout."""

import sys

if sys.version_info < (3, 12):
    raise SystemExit("Use Python 3.12 for the tested dependency set; see README.md")

from runtime import main
from experiment import run

if __name__ == "__main__":
    main(run)
