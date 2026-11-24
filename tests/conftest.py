import os
import sys
from pathlib import Path

os.environ.setdefault("LLM_BASE_URL", "http://unused")
os.environ.setdefault("LLM_API_KEY", "unused")
os.environ.setdefault("LLM_MODEL", "unused")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
