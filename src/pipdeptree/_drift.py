from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pipdeptree._models import PackageDAG

DriftKind = Literal["missing", "extra", "version_mismatch"]


@dataclass(frozen=True, order=True)
class DriftEntry:
    """A single per-package difference between the environment and a reference tree."""

    key: str
    package_name: str
    kind: DriftKind
    env_version: str | None = None
    ref_version: str | None = None
    dependents: tuple[str, ...] = ()


@dataclass
class DriftReport:
    """Aggregated drift between two dependency trees."""

    missing: list[DriftEntry] = field(default_factory=list)
    extra: list[DriftEntry] = field(default_factory=list)
    version_mismatch: list[DriftEntry] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.missing or self.extra or self.version_mismatch)

    def as_dict(self) -> dict[str, list[dict[str, object]]]:
        return {
            "missing": [_entry_dict(e) for e in self.missing],
            "extra": [_entry_dict(e) for e in self.extra],
            "version_mismatch": [_entry_dict(e) for e in self.version_mismatch],
        }


def _entry_dict(entry: DriftEntry) -> dict[str, object]:
    d: dict[str, object] = {"key": entry.key, "package_name": entry.package_name}
    if entry.env_version is not None:
        d["env_version"] = entry.env_version
    if entry.ref_version is not None:
        d["ref_version"] = entry.ref_version
    if entry.dependents:
        d["dependents"] = list(entry.dependents)
    return d


def compute_drift(env_tree: PackageDAG, ref_tree: PackageDAG) -> DriftReport:
    """Compare an installed environment tree against a reference (lock/index) tree."""
    env_index: dict[str, tuple[str, str]] = {}
    for pkg in env_tree:
        env_index[pkg.key] = (pkg.project_name, pkg.version)

    ref_index: dict[str, tuple[str, str]] = {}
    for pkg in ref_tree:
        ref_index[pkg.key] = (pkg.project_name, pkg.version)

    ref_dependents = _build_dependents_map(ref_tree)

    env_keys = set(env_index)
    ref_keys = set(ref_index)

    missing: list[DriftEntry] = []
    for key in sorted(ref_keys - env_keys):
        name, version = ref_index[key]
        missing.append(
            DriftEntry(
                key=key,
                package_name=name,
                kind="missing",
                ref_version=version,
                dependents=tuple(sorted(ref_dependents.get(key, ()))),
            ),
        )

    extra: list[DriftEntry] = []
    for key in sorted(env_keys - ref_keys):
        name, version = env_index[key]
        extra.append(
            DriftEntry(
                key=key,
                package_name=name,
                kind="extra",
                env_version=version,
            ),
        )

    version_mismatch: list[DriftEntry] = []
    for key in sorted(env_keys & ref_keys):
        env_name, env_ver = env_index[key]
        ref_name, ref_ver = ref_index[key]
        if env_ver != ref_ver:
            version_mismatch.append(
                DriftEntry(
                    key=key,
                    package_name=ref_name,
                    kind="version_mismatch",
                    env_version=env_ver,
                    ref_version=ref_ver,
                    dependents=tuple(sorted(ref_dependents.get(key, ()))),
                ),
            )

    return DriftReport(missing=missing, extra=extra, version_mismatch=version_mismatch)


def _build_dependents_map(tree: PackageDAG) -> dict[str, set[str]]:
    """Build a reverse lookup: for each package key, which parent packages require it."""
    dependents: dict[str, set[str]] = {}
    for parent, children in tree.items():
        for child in children:
            dependents.setdefault(child.key, set()).add(parent.project_name)
    return dependents


__all__ = [
    "DriftEntry",
    "DriftReport",
    "compute_drift",
]
