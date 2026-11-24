import subprocess
import sys


def test_cli_value_runs():
    result = subprocess.run([sys.executable, "-m", "inventory.cli", "value", "data/items.csv"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip().replace(".", "", 1).isdigit()
