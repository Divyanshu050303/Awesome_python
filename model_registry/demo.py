"""Walkthrough of the registry. Run: python demo.py"""

from pathlib import Path

from registry import (
    ModelInfo, ModelRegistry, Version, Stage,
    JSONStorage, MemoryStorage,
    ModelNotFound, VersionExists, InvalidVersion,
)

LINE = "-" * 68


def header(text):
    print(f"\n{LINE}\n{text}\n{LINE}")


def main():
    reg = ModelRegistry()          # defaults to MemoryStorage

    # ---------------------------------------------------------------- 1
    header("1. REGISTRATION")

    reg.register(ModelInfo(
        name="Sentiment BERT", version="1.0.0", framework="PyTorch",
        parameters=110_000_000, tags={"NLP", "classification"},
        metrics={"accuracy": 0.912, "f1": 0.908},
        description="Baseline sentiment classifier",
    ))
    reg.register(ModelInfo(
        name="sentiment-bert", version="1.1.0", framework="pytorch",
        parameters=110_000_000, tags={"nlp", "classification"},
        metrics={"accuracy": 0.934, "f1": 0.931},
        description="Retrained on the 2026 corpus",
    ))
    reg.register(ModelInfo(
        name="sentiment-bert", version="2.0.0", framework="pytorch",
        parameters=340_000_000, tags={"nlp", "classification", "large"},
        metrics={"accuracy": 0.951, "f1": 0.948},
    ))
    reg.register(ModelInfo(
        name="churn-xgb", version="0.3.1", framework="xgboost",
        parameters=45_000, tags={"tabular"},
        metrics={"auc": 0.872},
    ))
    reg.register(ModelInfo(
        name="image-clip", version="1.4.0", framework="pytorch",
        parameters=630_000_000, tags={"vision", "embedding"},
        metrics={"recall@5": 0.887},
    ))

    print(reg)
    print("\nnote normalisation: 'Sentiment BERT' + 'PyTorch' + {'NLP'} became")
    print("  ", reg["sentiment-bert", "1.0.0"].name,
          "/", reg["sentiment-bert", "1.0.0"].framework,
          "/", sorted(reg["sentiment-bert", "1.0.0"].tags))

    # ---------------------------------------------------------------- 2
    header("2. VERSIONING")

    print("versions of sentiment-bert:")
    for m in reg.versions("sentiment-bert"):
        print("   ", m)

    print("\nget() with no version returns the highest:", reg.get("sentiment-bert").version)
    print("Version sorts numerically, not as text:")
    print("   ", sorted([Version.parse("1.10.0"), Version.parse("1.9.0"), Version.parse("1.2.0")],
                        key=str), " <- as strings")
    print("   ", [str(v) for v in sorted([Version.parse("1.10.0"),
                                          Version.parse("1.9.0"),
                                          Version.parse("1.2.0")])], " <- as Versions")
    print("\nbump():", Version.parse("2.0.0"), "->", Version.parse("2.0.0").bump("minor"))

    # ---------------------------------------------------------------- 3
    header("3. METADATA + LIFECYCLE")

    reg.promote("sentiment-bert", "1.1.0", Stage.PRODUCTION)
    reg.promote("sentiment-bert", "2.0.0", Stage.STAGING)
    reg.promote("churn-xgb", "0.3.1", Stage.PRODUCTION)

    for m in reg.versions("sentiment-bert"):
        print("   ", m)

    print("\npromoting 2.0.0 to production auto-archives 1.1.0:")
    reg.promote("sentiment-bert", "2.0.0", Stage.PRODUCTION)
    for m in reg.versions("sentiment-bert"):
        print("   ", m)

    print("\nlatest production version:", reg.latest("sentiment-bert", Stage.PRODUCTION))

    # ---------------------------------------------------------------- 4
    header("4. SEARCH")

    print("framework=pytorch, >100M params, sorted by size:")
    for m in reg.search(framework="pytorch", min_parameters=100_000_000,
                        sort_by="parameters"):
        print(f"    {m.name:<16} {m.version}  {m.size_label:>7}")

    print("\ntagged {nlp} with accuracy >= 0.93:")
    for m in reg.search(tags={"nlp"}, metric="accuracy", min_metric=0.93,
                        sort_by="metric"):
        print(f"    {m.name}@{m.version}  accuracy={m.metrics['accuracy']}")

    print("\neverything in production:")
    for m in reg.search(stage=Stage.PRODUCTION):
        print("   ", m)

    print("\nfree-text 'clip':", [str(m) for m in reg.search("clip")])

    # ---------------------------------------------------------------- 5
    header("5. DIFF + STATS")

    print("sentiment-bert 1.0.0 vs 2.0.0:")
    for metric, (a, b) in reg.diff("sentiment-bert", "1.0.0", "2.0.0").items():
        delta = f"{b - a:+.3f}" if a is not None and b is not None else "n/a"
        print(f"    {metric:<10} {a}  ->  {b}   ({delta})")

    print("\nstats:")
    for k, v in reg.stats().items():
        print(f"    {k}: {v}")

    # ---------------------------------------------------------------- 6
    header("6. DUNDER PROTOCOL")

    print("len(reg):", len(reg))
    print("'churn-xgb' in reg:", "churn-xgb" in reg)
    print("'nonexistent' in reg:", "nonexistent" in reg)
    print("reg['image-clip']:", reg["image-clip"])
    print("reg['sentiment-bert', '1.0.0']:", reg["sentiment-bert", "1.0.0"])
    print("iterating gives every version:", len(list(reg)))

    # ---------------------------------------------------------------- 7
    header("7. IMMUTABILITY")

    m = reg.get("churn-xgb")
    try:
        m.parameters = 999
    except Exception as e:
        print("direct mutation blocked:", type(e).__name__, "-", e)

    improved = m.with_metrics(auc=0.901)
    print("\noriginal metrics:", m.metrics)
    print("derived  metrics:", improved.metrics)
    print("original untouched:", m.metrics["auc"] == 0.872)
    print("equal anyway (metrics are payload, not identity):", m == improved)
    print("so both hash the same:", hash(m) == hash(improved))

    # ---------------------------------------------------------------- 8
    header("8. ERRORS")

    for label, fn in [
        ("unknown model",   lambda: reg.get("does-not-exist")),
        ("unknown version", lambda: reg.get("churn-xgb", "9.9.9")),
        ("duplicate",       lambda: reg.register(reg.get("churn-xgb"))),
        ("bad version",     lambda: Version.parse("1.x.3")),
        ("bad parameters",  lambda: ModelInfo(name="x", version="1.0.0",
                                              framework="f", parameters=-5)),
        ("wrong type",      lambda: reg.register({"name": "oops"})),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"    {label:<16} {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- 9
    header("9. DELETION")

    print("before:", reg.list_models())
    print("deleted", reg.delete("sentiment-bert", "1.0.0"), "version")
    print("remaining:", [str(m.version) for m in reg.versions("sentiment-bert")])
    print("deleted", reg.delete("image-clip"), "versions (whole model)")
    print("after:", reg.list_models())

    # ---------------------------------------------------------------- 10
    header("10. SWAPPABLE STORAGE")

    path = Path("models.json")
    with ModelRegistry(JSONStorage(path)) as disk:
        for m in reg:
            disk.register(m)
        print("registered", len(disk), "versions;", disk)
    print("context manager saved on exit ->", path.stat().st_size, "bytes")

    reloaded = ModelRegistry(JSONStorage(path))
    print("reloaded:", reloaded)
    print("round-trip preserved everything:", list(reloaded) == list(reg))

    print("\nsame registry class, different backend, zero code changes:")
    print("   ", MemoryStorage(), "|", JSONStorage(path))

    header("11. AUDIT LOG")
    for ts, action, detail in reg.audit_log[-6:]:
        print(f"    {ts:%H:%M:%S}  {action:<10} {detail}")

    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()