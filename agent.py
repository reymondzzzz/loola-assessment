"""Coding agent harness CLI. AGENT_MODE selects a preset from harness.config.MODES (default: hybrid)."""
import os
import sys

from harness.config import mode_config
from harness.graph import run_agent

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:])
    if not prompt.strip():
        print("No task provided. Exiting.")
        sys.exit(1)
    run_agent(prompt, mode_config(os.environ.get("AGENT_MODE", "lazy")))
