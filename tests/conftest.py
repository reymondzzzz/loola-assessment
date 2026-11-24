import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("LLM_BASE_URL", "http://unused")
os.environ.setdefault("LLM_API_KEY", "unused")
os.environ.setdefault("LLM_MODEL", "unused")
os.environ["AGENT_RUNS_DIR"] = tempfile.mkdtemp(prefix="agent-test-runs-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
