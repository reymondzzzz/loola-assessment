import subprocess
import sys


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "inventory.cli", *args], capture_output=True, text=True)


def test_header_only(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("name,price,qty\n")
    result = run_cli("value", str(path))
    assert result.returncode == 0 and result.stdout.strip() == "0.00"


def test_missing_file(tmp_path):
    result = run_cli("value", str(tmp_path / "nope.csv"))
    assert result.returncode == 1
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr
