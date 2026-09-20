"""
An ML model registry built out of the OOP toolkit:

  Version        frozen, ordered dataclass -- semantic versioning
  ModelInfo      frozen dataclass, validated in __post_init__
  Storage        Protocol -- polymorphic backends, injected not constructed
  ModelRegistry  composition + dunder methods + context manager
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Exceptions -- a shallow hierarchy so callers can catch broadly or precisely
# --------------------------------------------------------------------------

class RegistryError(Exception):
    """Base for everything this module raises."""


class ModelNotFound(RegistryError):
    pass


class VersionExists(RegistryError):
    pass


class InvalidVersion(RegistryError, ValueError):
    """Also a ValueError, so `except ValueError` still catches parse failures."""


# --------------------------------------------------------------------------
# Version
# --------------------------------------------------------------------------

@dataclass(frozen=True, order=True, slots=True)
class Version:
    """Semantic version. order=True gives <, >, sorting for free."""
    major: int
    minor: int = 0
    patch: int = 0

    @classmethod
    def parse(cls, text: str | "Version") -> "Version":
        if isinstance(text, Version):
            return text
        parts = str(text).strip().lstrip("v").split(".")
        if not 1 <= len(parts) <= 3:
            raise InvalidVersion(f"bad version: {text!r}")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            raise InvalidVersion(f"bad version: {text!r}") from None
        if any(n < 0 for n in nums):
            raise InvalidVersion(f"negative component in {text!r}")
        return cls(*nums)

    def bump(self, part: str = "patch") -> "Version":
        """Return the next version. Immutable objects derive, never mutate."""
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise ValueError("part must be major, minor or patch")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class Stage(str, Enum):
    """Subclassing str means these serialise to JSON as plain strings."""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


# --------------------------------------------------------------------------
# ModelInfo
# --------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class ModelInfo:
    """
    One immutable snapshot of one model version.

    Identity is (name, version, framework). Metrics, tags and metadata are
    payload -- they carry compare=False so the object stays hashable and so
    two records of the same version are equal regardless of annotations.
    """
    name: str
    version: Version
    framework: str
    parameters: int

    stage: Stage = Stage.DEV
    metrics: dict[str, float] = field(default_factory=dict, compare=False)
    tags: frozenset[str] = field(default_factory=frozenset, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    description: str = field(default="", compare=False)
    artifact_path: str = field(default="", compare=False)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc), compare=False
    )

    def __post_init__(self):
        # frozen=True blocks normal assignment, so normalise via object.__setattr__
        set_ = object.__setattr__

        if not self.name or not self.name.strip():
            raise ValueError("name must not be empty")
        set_(self, "name", self.name.strip().lower().replace(" ", "-"))

        set_(self, "version", Version.parse(self.version))
        set_(self, "stage", Stage(self.stage))
        set_(self, "framework", self.framework.strip().lower())
        set_(self, "tags", frozenset(t.strip().lower() for t in self.tags if t.strip()))

        if not isinstance(self.parameters, int) or self.parameters < 0:
            raise ValueError(f"parameters must be a non-negative int, got {self.parameters!r}")

    # -- derived, computed on access rather than stored -------------------

    @property
    def key(self) -> tuple[str, Version]:
        return (self.name, self.version)

    @property
    def size_label(self) -> str:
        n = self.parameters
        for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if n >= limit:
                return f"{n / limit:.1f}{suffix}"
        return str(n)

    @property
    def is_live(self) -> bool:
        return self.stage is Stage.PRODUCTION

    # -- immutable updates ------------------------------------------------

    def with_stage(self, stage: Stage) -> "ModelInfo":
        return replace(self, stage=Stage(stage))

    def with_metrics(self, **metrics: float) -> "ModelInfo":
        return replace(self, metrics={**self.metrics, **metrics})

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = str(self.version)
        d["stage"] = self.stage.value
        d["tags"] = sorted(self.tags)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInfo":
        d = dict(d)
        d["version"] = Version.parse(d["version"])
        d["stage"] = Stage(d["stage"])
        d["tags"] = frozenset(d.get("tags", ()))
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)

    def __str__(self) -> str:
        return f"{self.name}@{self.version} [{self.stage.value}] {self.size_label} params"


# --------------------------------------------------------------------------
# Storage -- Protocol, not a base class. Any object with these two methods
# works, including ones defined elsewhere that never heard of this module.
# --------------------------------------------------------------------------

@runtime_checkable
class Storage(Protocol):
    def load(self) -> list[dict]: ...
    def save(self, records: list[dict]) -> None: ...


class MemoryStorage:
    """Default backend. Nothing survives the process."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def load(self) -> list[dict]:
        return list(self._records)

    def save(self, records: list[dict]) -> None:
        self._records = list(records)

    def __repr__(self) -> str:
        return f"MemoryStorage({len(self._records)} records)"


class JSONStorage:
    """Persists to a JSON file. Same two methods, so it drops straight in."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return json.load(f)

    def save(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(records, f, indent=2)
        tmp.replace(self.path)      # atomic: never leaves a half-written file

    def __repr__(self) -> str:
        return f"JSONStorage({str(self.path)!r})"


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

class ModelRegistry:
    """
    Holds many versions of many models.

    Internal shape: {name: {Version: ModelInfo}}. That is an implementation
    detail behind a leading underscore -- callers go through the methods.
    """

    def __init__(self, storage: Storage | None = None, *, autoload: bool = True):
        # Injected, not constructed here: swap in JSONStorage or a fake
        # without touching this class.
        self._storage: Storage = storage if storage is not None else MemoryStorage()
        self._models: dict[str, dict[Version, ModelInfo]] = {}
        self._audit: list[tuple[datetime, str, str]] = []
        if autoload:
            self.load()

    # -- core operations ---------------------------------------------------

    def register(self, model: ModelInfo, *, overwrite: bool = False) -> ModelInfo:
        if not isinstance(model, ModelInfo):
            raise TypeError(f"expected ModelInfo, got {type(model).__name__}")

        versions = self._models.setdefault(model.name, {})
        if model.version in versions and not overwrite:
            raise VersionExists(
                f"{model.name}@{model.version} already registered "
                f"(pass overwrite=True to replace)"
            )
        action = "overwrite" if model.version in versions else "register"
        versions[model.version] = model
        self._log(action, f"{model.name}@{model.version}")
        return model

    def get(self, name: str, version: str | Version | None = None) -> ModelInfo:
        """Fetch one version. With no version, the highest."""
        name = name.strip().lower().replace(" ", "-")
        versions = self._models.get(name)
        if not versions:
            raise ModelNotFound(f"no model named {name!r}")

        if version is None:
            return versions[max(versions)]

        v = Version.parse(version)
        if v not in versions:
            available = ", ".join(str(x) for x in sorted(versions))
            raise ModelNotFound(f"{name} has no version {v} (have: {available})")
        return versions[v]

    def latest(self, name: str, stage: Stage | None = None) -> ModelInfo:
        """Highest version, optionally restricted to a stage."""
        candidates = self.versions(name)
        if stage is not None:
            candidates = [m for m in candidates if m.stage is Stage(stage)]
            if not candidates:
                raise ModelNotFound(f"{name} has no version in stage {Stage(stage).value}")
        return max(candidates, key=lambda m: m.version)

    def versions(self, name: str) -> list[ModelInfo]:
        name = name.strip().lower().replace(" ", "-")
        versions = self._models.get(name)
        if not versions:
            raise ModelNotFound(f"no model named {name!r}")
        return [versions[v] for v in sorted(versions)]

    def list_models(self, *, detailed: bool = False) -> list:
        """Names only by default; the latest record of each if detailed."""
        names = sorted(self._models)
        if not detailed:
            return names
        return [self.get(n) for n in names]

    def delete(self, name: str, version: str | Version | None = None) -> int:
        """Delete one version, or every version when version is None."""
        name = name.strip().lower().replace(" ", "-")
        versions = self._models.get(name)
        if not versions:
            raise ModelNotFound(f"no model named {name!r}")

        if version is None:
            count = len(versions)
            del self._models[name]
            self._log("delete", f"{name} (all {count} versions)")
            return count

        v = Version.parse(version)
        if v not in versions:
            raise ModelNotFound(f"{name} has no version {v}")
        del versions[v]
        if not versions:                    # drop the empty shell
            del self._models[name]
        self._log("delete", f"{name}@{v}")
        return 1

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str = "",
        *,
        framework: str | None = None,
        stage: Stage | None = None,
        tags: set[str] | None = None,
        min_parameters: int | None = None,
        max_parameters: int | None = None,
        metric: str | None = None,
        min_metric: float | None = None,
        latest_only: bool = False,
        sort_by: str = "name",
    ) -> list[ModelInfo]:
        """Every filter is optional and they compose with AND."""
        if latest_only:
            pool = [self.get(n) for n in self._models]
        else:
            pool = [m for vs in self._models.values() for m in vs.values()]

        q = query.strip().lower()
        results = []
        for m in pool:
            if q and q not in m.name and q not in m.description.lower():
                continue
            if framework and m.framework != framework.strip().lower():
                continue
            if stage is not None and m.stage is not Stage(stage):
                continue
            if tags and not {t.lower() for t in tags} <= m.tags:
                continue
            if min_parameters is not None and m.parameters < min_parameters:
                continue
            if max_parameters is not None and m.parameters > max_parameters:
                continue
            if metric is not None:
                value = m.metrics.get(metric)
                if value is None:
                    continue
                if min_metric is not None and value < min_metric:
                    continue
            results.append(m)

        keys = {
            "name": lambda m: (m.name, m.version),
            "version": lambda m: m.version,
            "parameters": lambda m: -m.parameters,
            "created": lambda m: m.created_at,
            "metric": lambda m: -(m.metrics.get(metric or "", 0.0)),
        }
        return sorted(results, key=keys.get(sort_by, keys["name"]))

    # -- lifecycle ---------------------------------------------------------

    def promote(self, name: str, version: str | Version, stage: Stage) -> ModelInfo:
        """
        Move a version to a stage. Promoting to PRODUCTION archives whatever
        was in production, so the invariant 'at most one live version' holds.
        """
        model = self.get(name, version)
        stage = Stage(stage)

        if stage is Stage.PRODUCTION:
            for other in self.versions(name):
                if other.is_live and other.version != model.version:
                    self._models[name][other.version] = other.with_stage(Stage.ARCHIVED)
                    self._log("archive", f"{name}@{other.version}")

        updated = model.with_stage(stage)
        self._models[name][model.version] = updated
        self._log("promote", f"{name}@{model.version} -> {stage.value}")
        return updated

    def diff(self, name: str, v1, v2) -> dict[str, tuple[float | None, float | None]]:
        """Metric-by-metric comparison of two versions."""
        a, b = self.get(name, v1), self.get(name, v2)
        keys = sorted(set(a.metrics) | set(b.metrics))
        return {k: (a.metrics.get(k), b.metrics.get(k)) for k in keys}

    def stats(self) -> dict[str, Any]:
        everything = list(self)
        by_framework: dict[str, int] = {}
        by_stage: dict[str, int] = {}
        for m in everything:
            by_framework[m.framework] = by_framework.get(m.framework, 0) + 1
            by_stage[m.stage.value] = by_stage.get(m.stage.value, 0) + 1
        return {
            "models": len(self._models),
            "versions": len(everything),
            "total_parameters": sum(m.parameters for m in everything),
            "by_framework": by_framework,
            "by_stage": by_stage,
        }

    @property
    def audit_log(self) -> list[tuple[datetime, str, str]]:
        return list(self._audit)          # a copy: callers can't corrupt it

    def _log(self, action: str, detail: str) -> None:
        self._audit.append((datetime.now(timezone.utc), action, detail))

    # -- persistence -------------------------------------------------------

    def save(self) -> int:
        records = [m.to_dict() for m in self]
        self._storage.save(records)
        return len(records)

    def load(self) -> int:
        count = 0
        for record in self._storage.load():
            model = ModelInfo.from_dict(record)
            self._models.setdefault(model.name, {})[model.version] = model
            count += 1
        return count

    def __enter__(self) -> "ModelRegistry":
        return self

    def __exit__(self, exc_type, exc_val, tb) -> bool:
        if exc_type is None:              # only persist a clean exit
            self.save()
        return False                      # never swallow the exception

    # -- dunder protocol ---------------------------------------------------

    def __len__(self) -> int:
        return sum(len(v) for v in self._models.values())

    def __iter__(self) -> Iterator[ModelInfo]:
        for name in sorted(self._models):
            for version in sorted(self._models[name]):
                yield self._models[name][version]

    def __contains__(self, item) -> bool:
        if isinstance(item, ModelInfo):
            return item.version in self._models.get(item.name, {})
        return str(item).strip().lower().replace(" ", "-") in self._models

    def __getitem__(self, key) -> ModelInfo:
        """registry['bert'] or registry['bert', '1.2.0']"""
        if isinstance(key, tuple):
            return self.get(*key)
        return self.get(key)

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry {len(self._models)} models / "
            f"{len(self)} versions, storage={self._storage!r}>"
        )