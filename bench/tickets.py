import csv, json, re, runpy, shutil, sqlite3, subprocess, sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Callable

from generate import march_revenue_ranking

BENCH_DIR = Path(__file__).resolve().parent
VISIBLE_TESTS_DIR = BENCH_DIR / "visible_tests"
HIDDEN_TESTS_DIR = BENCH_DIR / "hidden_tests"
RELEASE_NAME = "ORCHID-7"
LOG_ENTRY = re.compile(r"^\d{4}-\d\d-\d\dT\S+ (\w+) \[([\w-]+)\] ")

Check = Callable[[Path, dict], tuple[bool, str]]


@dataclass
class Ticket:
    text: str
    visible_test: str | None
    check: Check


def run_pytest(workdir: Path, *test_files: str) -> tuple[bool, str]:
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files],
                            cwd=workdir, capture_output=True, text=True, timeout=180)
    return result.returncode == 0, "\n".join((result.stdout + result.stderr).splitlines()[-25:])


def pytest_check(visible: str, hidden: str) -> Check:
    return lambda workdir, state: run_pytest(workdir, f"tests/{visible}", str(HIDDEN_TESTS_DIR / hidden))


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def expected_error_counts(workdir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in (workdir / "logs" / "app.log").read_text().splitlines():
        match = LOG_ENTRY.match(line)
        if match and match.group(1) == "ERROR":
            counts[match.group(2)] = counts.get(match.group(2), 0) + 1
    return counts


def check_report(workdir: Path, state: dict) -> tuple[bool, str]:
    ok = read_json(workdir / "report.json") == expected_error_counts(workdir)
    return ok, "" if ok else "report.json is missing, not valid JSON, or has wrong counts."


def check_primary_service(workdir: Path, state: dict) -> tuple[bool, str]:
    counts = expected_error_counts(workdir)
    expected = min(service for service, count in counts.items() if count == max(counts.values()))
    try:
        actual = runpy.run_path(str(workdir / "inventory" / "config.py")).get("PRIMARY_SERVICE")
    except Exception as error:
        return False, f"inventory/config.py failed to load: {error}"
    return actual == expected, "" if actual == expected else f"PRIMARY_SERVICE is {actual!r}, which is not correct."


def read_items(workdir: Path) -> list[dict]:
    with open(workdir / "data" / "items.csv", newline="", encoding="utf-8-sig") as f:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]


def check_cli(workdir: Path, state: dict) -> tuple[bool, str]:
    ok, output = run_pytest(workdir, "tests/test_cli.py", str(HIDDEN_TESTS_DIR / "test_t5_cli.py"))
    if not ok:
        return False, output
    expected = sum((Decimal(row["price"]) * int(row["qty"]) for row in read_items(workdir)), Decimal(0))
    expected_text = f"{expected.quantize(Decimal('0.01'), ROUND_HALF_UP)}"
    actual = subprocess.run([sys.executable, "-m", "inventory.cli", "value", "data/items.csv"],
                            cwd=workdir, capture_output=True, text=True).stdout.strip()
    return actual == expected_text, "" if actual == expected_text else f"Printed {actual!r}, which is not the correct total."


def check_top_customers(workdir: Path, state: dict) -> tuple[bool, str]:
    db = sqlite3.connect(workdir / "data" / "shop.db")
    names = dict(db.execute("SELECT id, name FROM customers"))
    expected = [{"customer_id": cid, "name": names[cid], "revenue": float(revenue.quantize(Decimal("0.01")))}
                for cid, revenue in march_revenue_ranking(db)[:3]]
    db.close()
    actual = read_json(workdir / "top_customers.json")
    ok = isinstance(actual, list) and len(actual) == 3 and all(
        isinstance(a, dict) and a.get("customer_id") == e["customer_id"] and a.get("name") == e["name"]
        and isinstance(a.get("revenue"), (int, float)) and round(a["revenue"], 2) == e["revenue"]
        for a, e in zip(actual, expected))
    return ok, "" if ok else "top_customers.json is missing or incorrect."


def check_restock_suppliers(workdir: Path, state: dict) -> tuple[bool, str]:
    stock: dict[str, int] = {}
    for row in read_items(workdir):
        stock[row["name"]] = stock.get(row["name"], 0) + int(row["qty"])
    low = {name.casefold() for name, qty in stock.items() if qty <= 5}
    suppliers = json.loads((workdir / "data" / "suppliers.json").read_text())["suppliers"]
    expected = sorted(s["id"] for s in suppliers
                      if s["active"] and any(entry["sku"].casefold() in low for entry in s.get("catalog", [])))
    ok = read_json(workdir / "restock_suppliers.json") == expected
    return ok, "" if ok else "restock_suppliers.json is missing or incorrect."


def check_bad_commit(workdir: Path, state: dict) -> tuple[bool, str]:
    path = workdir / "BAD_COMMIT"
    written = path.read_text().strip() if path.exists() else ""
    if len(written) < 7 or not state["bad_commit"].startswith(written):
        return False, "BAD_COMMIT is missing or names the wrong commit."
    return run_pytest(workdir, "tests/test_tax.py", str(HIDDEN_TESTS_DIR / "test_t10_tax.py"))


def check_release(workdir: Path, state: dict) -> tuple[bool, str]:
    path = workdir / "RELEASE"
    ok = path.exists() and path.read_text().strip() == RELEASE_NAME
    return ok, "" if ok else "RELEASE is missing or has the wrong code name."


TICKETS = [
    Ticket("""Pricing bugs. tests/test_pricing.py fails. Fix inventory/pricing.py so it behaves exactly as its
docstrings describe (edge cases are checked too).
Note for later: this sprint's release code name is ORCHID-7. A later ticket needs it; it will not be repeated.""",
           "test_pricing.py", pytest_check("test_pricing.py", "test_t1_pricing.py")),
    Ticket("""Store bugs. tests/test_store.py fails. Fix inventory/models.py and inventory/store.py so Item and Store
behave exactly as their docstrings describe (edge cases are checked too).""",
           "test_store.py", pytest_check("test_store.py", "test_t2_store.py")),
    Ticket("""Error report. Count the ERROR-level entries per service in logs/app.log and write report.json: a JSON
object mapping service name to count, only for services with at least one ERROR entry.
An entry is a line that starts with a timestamp, followed by the level and the [service]. Lines that do not
start with a timestamp are continuations of the previous entry, not separate entries.""",
           None, check_report),
    Ticket("""Primary service. In inventory/config.py set PRIMARY_SERVICE to the name (a string) of the service with the
most ERROR entries in logs/app.log, counted as in the previous ticket. On a tie, pick the alphabetically first.""",
           None, check_primary_service),
    Ticket("""CLI. Add inventory/cli.py so that `python -m inventory.cli value <csv>` prints the total inventory value:
the sum of price * qty over all rows, formatted with exactly two decimals (e.g. 1234.50).
The CSV has a header row (name,price,qty); names may contain commas and quotes (standard CSV quoting); fields may
have surrounding spaces; blank lines are ignored; the file may start with a UTF-8 BOM. A file with only a header
prints 0.00. If the file does not exist, print an error message to stderr and exit with code 1 (no traceback).""",
           "test_cli.py", check_cli),
    Ticket("""Top customers. data/shop.db is a SQLite database (customers, orders, order_items). Write
top_customers.json: the 3 customers with the highest revenue in March 2026 (UTC), highest first, as a list of
{"customer_id": int, "name": str, "revenue": float}.
Revenue = sum of qty * unit_price over order_items of orders with status 'paid'. orders.created_at is ISO-8601
text and may carry a UTC offset. Round revenue to 2 decimals; on equal revenue the smaller customer_id comes first.
(FYI: old dashboards still show the retired code name ORCHID-5; it is obsolete.)""",
           None, check_top_customers),
    Ticket("""Restock suppliers. Load data/items.csv into a Store (rows with the same name are merged, as Store.add
does). Low-stock items are the names Store.low_stock(5) returns. data/suppliers.json lists suppliers; catalog SKUs
correspond to item names but are compared case-insensitively; some suppliers have no catalog.
Write restock_suppliers.json: the sorted list of ids of active suppliers that carry at least one low-stock item.""",
           None, check_restock_suppliers),
    Ticket("""Rename. Rename the Item attribute `qty` to `quantity` everywhere in the inventory package. Keep all existing
behaviour and every earlier ticket working. Data files are not part of the package: the CSV column stays `qty`,
and so do the "qty" keys of the dict rows used in legacy_report.py.""",
           "test_rename.py", pytest_check("test_rename.py", "test_t8_rename.py")),
    Ticket("""Legacy report. tests/test_legacy.py fails: build_report in inventory/legacy_report.py loses rows. Find and fix
the root cause. paginate(rows, page_size) must return pages of page_size with a possibly shorter last page, [] for
no rows, and raise ValueError when page_size < 1.""",
           "test_legacy.py", pytest_check("test_legacy.py", "test_t9_paginate.py")),
    Ticket("""Tax regression. compute_tax in inventory/tax.py gives wrong results for some countries; it used to be correct.
Use the git history to find the commit that introduced the regression and write its full hash to a file named
BAD_COMMIT. Then fix the bug, keeping the rates other commits added. Tax is rounded to cents, half up; unknown
countries raise KeyError.""",
           "test_tax.py", check_bad_commit),
    Ticket("""Release. Write this sprint's release code name (given in an earlier ticket) to a file named RELEASE,
containing just the name.""",
           None, check_release),
]


def issue(workdir: Path, number: int) -> None:
    ticket = TICKETS[number]
    if ticket.visible_test:
        shutil.copy(VISIBLE_TESTS_DIR / ticket.visible_test, workdir / "tests" / ticket.visible_test)
    print(f"TICKET {number + 1}/{len(TICKETS)}\n{ticket.text}")


def main() -> None:
    state_path, command = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "next"
    state = json.loads(state_path.read_text())
    workdir, current = Path(state["workdir"]), state["current"]
    if command == "current":
        print(f"TICKET {current + 1}/{len(TICKETS)}\n{TICKETS[current].text}" if 0 <= current < len(TICKETS) else "No open ticket.")
        return
    if current >= len(TICKETS):
        print("All tickets done.")
        return
    if current >= 0:
        for number in range(current, -1, -1):
            ok, message = TICKETS[number].check(workdir, state)
            if not ok:
                status = "is not done yet" if number == current else "was done earlier but is broken now"
                print(f"✗ Ticket {number + 1} {status}.\n{message}")
                sys.exit(1)
        print(f"✓ Ticket {current + 1} done.")
        state["closed"] = current + 1
    state["current"] = current + 1
    state_path.write_text(json.dumps(state))
    if state["current"] < len(TICKETS):
        issue(workdir, state["current"])
    else:
        print("All tickets done.")


if __name__ == "__main__":
    main()
