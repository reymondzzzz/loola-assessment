import json, os, random, shutil, sqlite3, subprocess, tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixture"

SERVICES = ["auth", "auth-legacy", "payments", "search", "orders", "gateway", "billing"]
NEVER_ERRORS = "billing"
LEVELS = ["DEBUG", "INFO", "INFO", "INFO", "WARN", "ERROR"]
MESSAGES = ["request handled", "cache miss", "retrying after upstream ERROR", "slow query detected",
            "connection reset", "user not found", "ERROR budget at 80%", "payload validated"]


def log_entry(timestamp: datetime, level: str, service: str, rng: random.Random) -> str:
    return f"{timestamp:%Y-%m-%dT%H:%M:%SZ} {level} [{service}] {rng.choice(MESSAGES)} id={rng.randint(1000, 9999)}"


def write_log(path: Path, rng: random.Random, entries: int = 4000) -> None:
    timestamp = datetime(2026, 3, 1, tzinfo=timezone.utc)
    lines: list[str] = []
    errors = {service: 0 for service in SERVICES}
    for _ in range(entries):
        timestamp += timedelta(seconds=rng.randint(1, 30))
        service = rng.choice(SERVICES)
        level = rng.choice(LEVELS)
        if service == NEVER_ERRORS and level == "ERROR":
            level = "INFO"
        errors[service] += level == "ERROR"
        line = log_entry(timestamp, level, service, rng)
        lines.append(line + "\r" if rng.random() < 0.05 else line)
        if level == "ERROR" and rng.random() < 0.3:
            lines += ["Traceback (most recent call last):",
                      f'  File "{service}/handler.py", line {rng.randint(10, 300)}, in handle',
                      "RuntimeError: ERROR [payments] upstream unavailable"]
    first, second = sorted(rng.sample([s for s in SERVICES if s != NEVER_ERRORS], 2))
    tie_count = max(errors.values()) + 3
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    lines.insert(0, log_entry(start - timedelta(minutes=5), "ERROR", second, rng))
    for service in (first, second):
        for _ in range(tie_count - errors[service] - (service == second)):
            lines.insert(rng.randint(1, len(lines) - 1), log_entry(start + timedelta(minutes=rng.randint(0, 10000)), "ERROR", service, rng))
    path.write_text("\n".join(lines) + "\n\n", newline="")


def write_items_csv(path: Path, rng: random.Random, rows: int = 200) -> list[str]:
    names = []
    lines = ["﻿name,price,qty"]
    for i in range(rows):
        if i % 7 == 0:
            name = f"widget {i}, size {rng.choice('SML')}"
            cell = f'"{name}"'
        elif i % 31 == 0:
            name, cell = f'bolt "heavy" {i}', f'"bolt ""heavy"" {i}"'
        else:
            name = cell = f"part-{i}"
        names.append(name)
        qty = rng.randint(0, 8) if rng.random() < 0.2 else rng.randint(0, 500)
        price = f"{rng.randint(50, 99999) / 100:.2f}"
        lines.append(f"{cell}, {price} ,{qty}" if i % 11 == 0 else f"{cell},{price},{qty}")
        if i % 40 == 0:
            lines.append("")
    for name in rng.sample([n for n in names if n.startswith("part-")], 12):
        lines.append(f"{name},{rng.randint(50, 9999) / 100:.2f},{rng.randint(1, 4)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return names


def write_suppliers(path: Path, rng: random.Random, item_names: list[str]) -> None:
    suppliers = []
    for i in range(1, 81):
        supplier = {"id": f"S-{i:03d}", "name": f"Supplier {i}", "active": rng.random() > 0.2,
                    "contact": {"email": f"s{i}@example.com", "region": rng.choice(["EU", "US", "APAC"])}}
        if rng.random() > 0.1:
            supplier["catalog"] = [{"sku": rng.choice([str.upper, str.lower, str.title])(sku) if rng.random() < 0.3 else sku,
                                    "lead_days": rng.randint(1, 40)}
                                   for sku in rng.sample(item_names, rng.randint(2, 7))]
        suppliers.append(supplier)
    path.write_text(json.dumps({"generated": "2026-03-31", "suppliers": suppliers}, indent=2))


def write_shop_db(path: Path, rng: random.Random) -> None:
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                             created_at TEXT NOT NULL, status TEXT NOT NULL);
        CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id), sku TEXT NOT NULL,
                                  qty INTEGER NOT NULL, unit_price REAL NOT NULL);
    """)
    customer_names = [f"Customer {i}" for i in range(1, 41)]
    customer_names[7] = customer_names[23] = "Alex Kim"
    db.executemany("INSERT INTO customers VALUES (?, ?)", list(enumerate(customer_names, start=1)))
    offsets = ["Z", "Z", "Z", "+02:00", "-05:00"]
    start = datetime(2026, 2, 20, tzinfo=timezone.utc)

    def add_order(customer_id: int, created_at: str, status: str, items: list[tuple[int, float]]) -> None:
        order_id = db.execute("INSERT INTO orders (customer_id, created_at, status) VALUES (?, ?, ?)",
                              (customer_id, created_at, status)).lastrowid
        db.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)",
                       [(order_id, f"sku-{rng.randint(1, 300)}", qty, price) for qty, price in items])

    for _ in range(500):
        moment = start + timedelta(minutes=rng.randint(0, 50 * 24 * 60))
        offset = rng.choice(offsets)
        created_at = moment.strftime("%Y-%m-%dT%H:%M:%SZ") if offset == "Z" else \
            moment.astimezone(timezone(timedelta(hours=int(offset[:3])))).isoformat()
        status = rng.choices(["paid", "refunded", "pending"], [75, 15, 10])[0]
        add_order(rng.randint(1, 40), created_at, status,
                  [(rng.randint(1, 5), rng.randint(100, 20000) / 100) for _ in range(rng.randint(1, 4))])
    for created_at, inside in [("2026-03-01T00:00:00Z", True), ("2026-03-31T23:59:59Z", True),
                               ("2026-04-01T00:00:00Z", False), ("2026-04-01T01:30:00+02:00", True),
                               ("2026-03-01T00:30:00+02:00", False)]:
        add_order(rng.randint(1, 40), created_at, "paid", [(1, 150.0)] if inside else [(10, 400.0)])
    db.commit()
    ranking = march_revenue_ranking(db)
    (_, third_revenue), (fourth_id, fourth_revenue) = ranking[2], ranking[3]
    add_order(fourth_id, "2026-03-15T12:00:00Z", "paid", [(1, float(third_revenue - fourth_revenue))])
    db.commit()
    db.close()


def march_revenue_ranking(db: sqlite3.Connection) -> list[tuple[int, Decimal]]:
    revenue: dict[int, Decimal] = {}
    for customer_id, created_at, qty, unit_price in db.execute(
            "SELECT o.customer_id, o.created_at, i.qty, i.unit_price FROM orders o "
            "JOIN order_items i ON i.order_id = o.id WHERE o.status = 'paid'"):
        moment = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(timezone.utc)
        if moment.year == 2026 and moment.month == 3:
            revenue[customer_id] = revenue.get(customer_id, Decimal(0)) + Decimal(str(unit_price)) * qty
    return sorted(revenue.items(), key=lambda pair: (-pair[1], pair[0]))


def write_legacy_report(path: Path, rng: random.Random) -> None:
    blocks = ['"""Legacy reporting helpers kept for the old dashboard. Do not delete without checking with ops."""\n']
    for i in range(80):
        weight = rng.randint(1, 9) / 10
        blocks.append(f'''
def summarize_metric_{i}(rows):
    """Aggregate metric {i} for dashboard panel {i}."""
    total = 0.0
    count = 0
    for row in rows:
        value = row.get("metric_{i}", 0)
        if value is None:
            continue
        total += value * {weight}
        count += 1
    if count == 0:
        return {{"metric": "metric_{i}", "average": 0.0, "count": 0}}
    return {{"metric": "metric_{i}", "average": round(total / count, {i % 4}), "count": count}}
''')
        if i == 45:
            blocks.append('''
def paginate(rows, page_size):
    """Split rows into pages of page_size; the last page may be shorter."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    pages = []
    for start in range(0, len(rows) - 1, page_size):
        pages.append(rows[start:start + page_size])
    return pages
''')
    blocks.append('''
def format_row(row):
    return f'{row["name"]:<20} {row["qty"]:>6} {row["price"]:>10.2f}'


def build_report(rows, page_size=20):
    lines = []
    for number, page in enumerate(paginate(rows, page_size), start=1):
        lines.append(f"== page {number} ==")
        lines.extend(format_row(row) for row in page)
    return "\\n".join(lines)
''')
    path.write_text("".join(blocks))


TAX_HEADER = "from decimal import ROUND_HALF_UP, Decimal\n\n"
TAX_BODY = '''

def compute_tax(amount: float, country: str) -> float:{doc}
    rate = Decimal(str(RATES[country]))
    return float((Decimal(str(amount)) * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
'''
TAX_DOC = '\n    """Tax for amount in country, rounded to cents half up. Unknown country raises KeyError."""'


def tax_source(rates: dict[str, float], documented: bool) -> str:
    return TAX_HEADER + f"RATES = {json.dumps(rates)}\n" + TAX_BODY.format(doc=TAX_DOC if documented else "")


def git(workdir: Path, *args: str, day: int = 1) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "Dev", "GIT_AUTHOR_EMAIL": "dev@example.com",
           "GIT_COMMITTER_NAME": "Dev", "GIT_COMMITTER_EMAIL": "dev@example.com",
           "GIT_AUTHOR_DATE": f"2026-02-{day:02d}T10:00:00Z", "GIT_COMMITTER_DATE": f"2026-02-{day:02d}T10:00:00Z"}
    return subprocess.run(["git", *args], cwd=workdir, env=env, check=True, capture_output=True, text=True).stdout.strip()


def write_git_history(workdir: Path) -> str:
    rates = {"DE": 0.19, "FR": 0.2, "GB": 0.2, "US": 0.0}
    tax_path = workdir / "inventory" / "tax.py"
    changes_path = workdir / "CHANGES.md"
    tax_path.write_text(tax_source(rates, documented=False))
    changes_path.write_text("# Changes\n")
    git(workdir, "init", "-q", "-b", "main")
    git(workdir, "add", "-A")
    git(workdir, "commit", "-q", "-m", "Initial import")
    bad_commit = ""
    for day in range(2, 22):
        if day == 12:
            rates = {**rates, "ES": 0.21, "GB": 0.02}
            tax_path.write_text(tax_source(rates, documented=False))
            message = "Add ES tax rate"
        elif day == 15:
            rates = {**rates, "NL": 0.21}
            tax_path.write_text(tax_source(rates, documented=False))
            message = "Add NL tax rate"
        elif day == 18:
            tax_path.write_text(tax_source(rates, documented=True))
            message = "Document compute_tax"
        else:
            message = f"Update changelog for build {day}"
        changes_path.write_text(changes_path.read_text() + f"- build {day}: {message}\n")
        git(workdir, "add", "-A")
        git(workdir, "commit", "-q", "-m", message, day=day)
        if day == 12:
            bad_commit = git(workdir, "rev-parse", "HEAD")
    return bad_commit


def make_workdir(seed: int) -> tuple[Path, str]:
    workdir = Path(tempfile.mkdtemp(prefix=f"bench-{seed}-"))
    shutil.copytree(FIXTURE_DIR, workdir, dirs_exist_ok=True)
    rng = random.Random(seed)
    for folder in ("logs", "data", "tests"):
        (workdir / folder).mkdir()
    write_log(workdir / "logs" / "app.log", rng)
    item_names = write_items_csv(workdir / "data" / "items.csv", rng)
    write_suppliers(workdir / "data" / "suppliers.json", rng, item_names)
    write_shop_db(workdir / "data" / "shop.db", rng)
    write_legacy_report(workdir / "inventory" / "legacy_report.py", rng)
    (workdir / "tests" / ".gitkeep").touch()
    return workdir, write_git_history(workdir)
