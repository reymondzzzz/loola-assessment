"""Coding agent harness CLI.

New run:  agent.py "<task>"          (AGENT_MODE, AGENT_CONTEXT_WINDOW, AGENT_EVICTION_POLICY select the config)
Resume:   agent.py --resume RUN_ID   (continues a stopped or crashed run from its last checkpoint)
"""
import argparse
import os
import sys

from harness.config import mode_config
from harness.graph import resume_agent, run_agent


def new_run_config():
    window = os.environ.get("AGENT_CONTEXT_WINDOW")
    return mode_config(os.environ.get("AGENT_MODE", "lazy"), int(window) if window else None,
                       eviction_policy=os.environ.get("AGENT_EVICTION_POLICY", "default"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coding agent harness")
    parser.add_argument("task", nargs="*", help="task for a new run")
    parser.add_argument("--resume", metavar="RUN_ID", help="resume a run by id")
    args = parser.parse_args()
    if args.resume:
        resume_agent(args.resume)
    elif " ".join(args.task).strip():
        run_agent(" ".join(args.task), new_run_config())
    else:
        print("No task provided. Exiting.")
        sys.exit(1)
