"""Print Makefile help by parsing `target: ## description` comments.

Usage: python scripts/make_help.py [MAKEFILE_PATH]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

C_BLUE = "\033[36m"
C_RESET = "\033[0m"

pattern = re.compile(r"^([a-zA-Z0-9_-]+):.*?## (.*)$", re.M)

text = Path(sys.argv[1] if len(sys.argv) > 1 else "Makefile").read_text(encoding="utf-8")
for target, desc in pattern.findall(text):
    print(f"  {C_BLUE}{target:<26s}{C_RESET} {desc}")
