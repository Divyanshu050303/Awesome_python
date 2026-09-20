"""
Demo: stream a CSV of expenses through the pipeline framework.

Run:  python demo.py
"""

import csv
import logging
import random
import tracemalloc
from pathlib import Path

from logger_pipeline import (
    Pipeline,
    pipeline_step,
    process_records,
    batched,
    summary,
    reset_log,
    setup_logging,
)

DATA = Path("expenses.csv")
CATEGORIES = ["Food", "Travel", "Shopping", "Rent", "Utilities"]


def make_sample_data(rows: int = 200_000) -> None:
    """Write a CSV with some deliberately messy rows."""
    if DATA.exists():
        return
    random.seed(42)
    with DATA.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "amount"])
        for i in range(rows):
            category = random.choice(CATEGORIES)
            amount = round(random.uniform(10, 5000), 2)
            if i % 5000 == 0:          # blank amount
                amount = ""
            elif i % 7000 == 0:        # negative amount
                amount = -amount
            elif i % 9000 == 0:        # stray whitespace + case
                category = f"  {category.upper()}  "
            w.writerow([i, category, amount])


# --------------------------------------------------------------------------
# The steps
# --------------------------------------------------------------------------

@pipeline_step
def load_data(path=DATA):
    """Yield one dict per row. The file is never held in memory whole."""
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


@pipeline_step(name="clean_data")
def clean_data(rows):
    """Drop unparseable and negative amounts, normalise category names."""
    for row in rows:
        raw = row["amount"].strip()
        if not raw:
            continue
        try:
            amount = float(raw)
        except ValueError:
            continue
        if amount <= 0:
            continue
        yield {
            "id": int(row["id"]),
            "category": row["category"].strip().title(),
            "amount": amount,
        }


@pipeline_step
def transform_data(rows):
    """Add derived fields, one record at a time."""
    def enrich(record):
        return record | {
            "amount_inr": round(record["amount"] * 1.0, 2),
            "band": "high" if record["amount"] > 1000 else "low",
        }

    yield from process_records(rows, enrich)


@pipeline_step
def summarise(rows):
    """Consume the stream and aggregate. This is what pulls everything."""
    totals: dict[str, float] = {}
    count = 0
    for batch in batched(rows, 5_000):
        for r in batch:
            totals[r["category"]] = totals.get(r["category"], 0) + r["amount"]
            count += 1
    return {"records": count, "totals": totals}


@pipeline_step(name="flaky_step", on_error="skip")
def flaky_step(rows):
    """Deliberately blows up partway through, to show error handling."""
    for i, row in enumerate(rows):
        if i == 3:
            raise ValueError("simulated upstream failure")
        yield row


# --------------------------------------------------------------------------

def main() -> None:
    setup_logging(logging.INFO)
    make_sample_data()
    print(f"\ndataset: {DATA.stat().st_size / 1_000_000:.1f} MB on disk\n")

    # --- happy path -------------------------------------------------------
    tracemalloc.start()
    pipe = Pipeline(load_data, name="expenses") | clean_data | transform_data | summarise
    result = pipe.run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + summary())
    print(f"\nrecords kept: {result['records']:,}")
    for cat, total in sorted(result["totals"].items(), key=lambda x: -x[1]):
        print(f"  {cat:<12} {total:>14,.2f}")
    print(f"\npeak memory while streaming: {peak / 1_000_000:.2f} MB")

    # --- the eager equivalent, for contrast -------------------------------
    tracemalloc.start()
    with DATA.open(newline="") as f:
        everything = list(csv.DictReader(f))
    _, peak_eager = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"peak memory holding all {len(everything):,} rows: "
          f"{peak_eager / 1_000_000:.2f} MB")
    del everything

    # --- failure path -----------------------------------------------------
    print("\n--- failure handling (on_error='skip') ---")
    reset_log()
    pipe2 = Pipeline(load_data, name="broken") | clean_data | flaky_step | summarise
    result2 = pipe2.run()
    print("\n" + summary())
    print(f"\npipeline survived; partial result: {result2['records']} records")


if __name__ == "__main__":
    main()