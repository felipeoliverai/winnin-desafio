import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
