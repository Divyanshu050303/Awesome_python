"""
Tests. Run: python test_registry.py   (or: pytest test_registry.py)

Note FakeStorage: because ModelRegistry takes its backend as a constructor
argument, testing persistence needs no files, no mocks and no patching.
That is the practical payoff of dependency injection.
"""

from registry import (
    ModelInfo, ModelRegistry, Version, Stage, Storage,
    MemoryStorage, ModelNotFound, VersionExists, InvalidVersion,
)


class FakeStorage:
    """Satisfies the Storage Protocol without inheriting from anything."""

    def __init__(self, records=None):
        self.records = records or []
        self.save_calls = 0

    def load(self):
        return list(self.records)

    def save(self, records):
        self.records = list(records)
        self.save_calls += 1


def model(name="m", version="1.0.0", **kw):
    kw.setdefault("framework", "pytorch")
    kw.setdefault("parameters", 1000)
    return ModelInfo(name=name, version=version, **kw)


TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn


# -- Version ---------------------------------------------------------------

@test
def version_parses_short_forms():
    assert Version.parse("2") == Version(2, 0, 0)
    assert Version.parse("2.1") == Version(2, 1, 0)
    assert Version.parse("v2.1.3") == Version(2, 1, 3)
    assert Version.parse(Version(1, 0, 0)) == Version(1, 0, 0)

@test
def version_rejects_junk():
    for bad in ["", "1.x.3", "1.2.3.4", "-1.0.0", "abc"]:
        try:
            Version.parse(bad)
        except InvalidVersion:
            continue
        raise AssertionError(f"{bad!r} should have failed")

@test
def version_orders_numerically():
    assert Version.parse("1.9.0") < Version.parse("1.10.0")
    assert Version.parse("2.0.0") > Version.parse("1.99.99")
    assert max([Version.parse(v) for v in ["1.0.0", "0.9.9", "1.0.1"]]) == Version(1, 0, 1)

@test
def version_is_hashable_and_immutable():
    v = Version(1, 0, 0)
    assert {v, Version(1, 0, 0)} == {v}
    try:
        v.major = 2
    except Exception:
        return
    raise AssertionError("frozen dataclass should reject assignment")


# -- ModelInfo -------------------------------------------------------------

@test
def modelinfo_normalises():
    m = model(name="  My Model  ", framework="PyTorch ", tags={"NLP", " vision "})
    assert m.name == "my-model"
    assert m.framework == "pytorch"
    assert m.tags == frozenset({"nlp", "vision"})

@test
def modelinfo_validates():
    for kw in [{"name": "  "}, {"parameters": -1}, {"parameters": "big"}]:
        try:
            model(**kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} should have failed")

@test
def modelinfo_identity_ignores_payload():
    a = model(metrics={"acc": 0.9})
    b = model(metrics={"acc": 0.1}, description="different")
    assert a == b and hash(a) == hash(b)

@test
def modelinfo_derives_not_mutates():
    a = model(metrics={"acc": 0.9})
    b = a.with_metrics(f1=0.8)
    assert a.metrics == {"acc": 0.9}
    assert b.metrics == {"acc": 0.9, "f1": 0.8}
    assert a.with_stage(Stage.PRODUCTION).is_live and not a.is_live

@test
def modelinfo_round_trips():
    a = model(metrics={"acc": 0.9}, tags={"x"}, metadata={"n": 1})
    b = ModelInfo.from_dict(a.to_dict())
    assert a == b and a.metrics == b.metrics and a.tags == b.tags
    assert a.created_at == b.created_at

@test
def size_label_scales():
    assert model(parameters=630_000_000).size_label == "630.0M"
    assert model(parameters=7_000_000_000).size_label == "7.0B"
    assert model(parameters=500).size_label == "500"


# -- Registry --------------------------------------------------------------

@test
def register_and_get():
    r = ModelRegistry()
    r.register(model())
    assert r.get("m").version == Version(1, 0, 0)
    assert len(r) == 1

@test
def duplicate_rejected_unless_overwrite():
    r = ModelRegistry()
    r.register(model())
    try:
        r.register(model())
        raise AssertionError("should have raised")
    except VersionExists:
        pass
    r.register(model(parameters=2000), overwrite=True)
    assert r.get("m").parameters == 2000
    assert len(r) == 1

@test
def get_without_version_returns_highest():
    r = ModelRegistry()
    for v in ["1.0.0", "1.10.0", "1.9.0"]:
        r.register(model(version=v))
    assert r.get("m").version == Version(1, 10, 0)

@test
def missing_raises():
    r = ModelRegistry()
    for fn in [lambda: r.get("nope"), lambda: r.versions("nope"),
               lambda: r.delete("nope")]:
        try:
            fn()
            raise AssertionError("should have raised")
        except ModelNotFound:
            pass

@test
def delete_version_then_model():
    r = ModelRegistry()
    r.register(model(version="1.0.0"))
    r.register(model(version="2.0.0"))
    assert r.delete("m", "1.0.0") == 1
    assert len(r) == 1 and "m" in r
    assert r.delete("m") == 1
    assert "m" not in r and len(r) == 0

@test
def deleting_last_version_removes_the_name():
    r = ModelRegistry()
    r.register(model())
    r.delete("m", "1.0.0")
    assert r.list_models() == []

@test
def promote_keeps_one_production_version():
    r = ModelRegistry()
    for v in ["1.0.0", "2.0.0"]:
        r.register(model(version=v))
    r.promote("m", "1.0.0", Stage.PRODUCTION)
    r.promote("m", "2.0.0", Stage.PRODUCTION)
    live = [m for m in r.versions("m") if m.is_live]
    assert len(live) == 1 and live[0].version == Version(2, 0, 0)
    assert r.get("m", "1.0.0").stage is Stage.ARCHIVED

@test
def latest_by_stage():
    r = ModelRegistry()
    for v in ["1.0.0", "2.0.0"]:
        r.register(model(version=v))
    r.promote("m", "1.0.0", Stage.PRODUCTION)
    assert r.latest("m").version == Version(2, 0, 0)
    assert r.latest("m", Stage.PRODUCTION).version == Version(1, 0, 0)

@test
def search_filters_compose():
    r = ModelRegistry()
    r.register(model(name="a", framework="pytorch", parameters=100,
                     tags={"nlp"}, metrics={"acc": 0.9}))
    r.register(model(name="b", framework="xgboost", parameters=5000,
                     tags={"tabular"}, metrics={"acc": 0.5}))
    assert len(r.search(framework="pytorch")) == 1
    assert len(r.search(min_parameters=1000)) == 1
    assert len(r.search(tags={"nlp"})) == 1
    assert len(r.search(metric="acc", min_metric=0.8)) == 1
    assert len(r.search(framework="pytorch", min_parameters=1000)) == 0
    assert len(r.search()) == 2

@test
def search_skips_models_missing_the_metric():
    r = ModelRegistry()
    r.register(model(name="a", metrics={"acc": 0.9}))
    r.register(model(name="b"))
    assert [m.name for m in r.search(metric="acc", min_metric=0.0)] == ["a"]

@test
def latest_only_collapses_versions():
    r = ModelRegistry()
    for v in ["1.0.0", "2.0.0"]:
        r.register(model(version=v))
    assert len(r.search()) == 2
    assert len(r.search(latest_only=True)) == 1

@test
def dunders_work():
    r = ModelRegistry()
    r.register(model(version="1.0.0"))
    r.register(model(version="2.0.0"))
    assert len(r) == 2
    assert "m" in r and "zzz" not in r
    assert r["m"].version == Version(2, 0, 0)
    assert r["m", "1.0.0"].version == Version(1, 0, 0)
    assert [str(x.version) for x in r] == ["1.0.0", "2.0.0"]
    assert r.get("m") in r

@test
def register_rejects_wrong_type():
    r = ModelRegistry()
    try:
        r.register({"name": "m"})
        raise AssertionError("should have raised")
    except TypeError:
        pass


# -- Storage ---------------------------------------------------------------

@test
def fake_storage_satisfies_protocol():
    assert isinstance(FakeStorage(), Storage)
    assert isinstance(MemoryStorage(), Storage)

@test
def save_and_reload_round_trips():
    store = FakeStorage()
    r = ModelRegistry(store)
    r.register(model(metrics={"acc": 0.9}, tags={"nlp"}))
    r.save()
    assert store.save_calls == 1
    r2 = ModelRegistry(store)
    assert len(r2) == 1
    assert r2.get("m").metrics == {"acc": 0.9}

@test
def context_manager_saves_on_clean_exit():
    store = FakeStorage()
    with ModelRegistry(store) as r:
        r.register(model())
    assert store.save_calls == 1

@test
def context_manager_skips_save_on_error():
    store = FakeStorage()
    try:
        with ModelRegistry(store) as r:
            r.register(model())
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store.save_calls == 0      # and the exception was not swallowed

@test
def audit_log_is_a_copy():
    r = ModelRegistry()
    r.register(model())
    r.audit_log.clear()
    assert len(r.audit_log) == 1


# --------------------------------------------------------------------------

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  pass  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)