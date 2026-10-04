"""Allow `python -m shipdoc`."""

from __future__ import annotations

import sys

from shipdoc.cli import main

if __name__ == "__main__":
    sys.exit(main())
