"""Legacy entrypoint for LLM-as-judge eval. Prefer scripts/eval_llm_judge.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dpo_project.cli.eval_llm_judge import main


if __name__ == "__main__":
    main()
