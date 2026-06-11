"""
Read a PEP 751 lock file (``pylock.toml``) and render its dependency tree.

This is the engine behind the ``from-lock`` CLI subcommand. A PEP 751 lock is *already resolved* -- it carries the
pinned packages, their versions, and the forward edges between them -- so rendering it is purely a TOML to
:class:`~pipdeptree._models.PackageDAG` mapping. There is no resolver, no network, and no package index involved,
which is why ``from-lock`` works fully offline and needs no optional extra.

The lock's ``packages`` array becomes :class:`~pipdeptree._synthetic_dist.SyntheticDistribution` objects (the same
adapter the ``from-index`` path uses), so the existing DAG and every renderer work unchanged. Each package's
``dependencies`` array lists only child *names*; the child's version is looked up in the ``packages`` array by PEP 503
canonical name so an edge ``Foo_Bar`` matches a package ``foo-bar``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from packaging.utils import canonicalize_name

from pipdeptree._synthetic_dist import SyntheticDistribution

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    import tomllib
else:  # pragma: <3.11 cover
    import tomli as tomllib

if TYPE_CHECKING:
    from importlib.metadata import Distribution
    from pathlib import Path

    from packaging.utils import NormalizedName


class FromLockError(ValueError):
    """Raised when a PEP 751 lock file is missing or cannot be parsed into a dependency tree."""


def load_lock(path: Path) -> list[Distribution]:
    """
    Parse a PEP 751 ``pylock.toml`` into Distribution-like objects ready for the existing DAG pipeline.

    :param path: the ``pylock.toml`` lock file to read.
    :returns: a list of :class:`importlib.metadata.Distribution` look-alikes ready for
        :meth:`pipdeptree._models.PackageDAG.from_pkgs`.
    :raises FromLockError: if the file is missing, is not valid TOML, lacks a ``packages`` array, or has a package
        without a ``name``.
    """
    if not path.is_file():
        msg = f"lock file does not exist: {path}"
        raise FromLockError(msg)

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        msg = f"not a valid PEP 751 lock file: {path} (malformed TOML: {exc})"
        raise FromLockError(msg) from exc

    packages = data.get("packages")
    if not isinstance(packages, list):
        msg = f"not a valid PEP 751 lock file: {path} (missing 'packages' array)"
        raise FromLockError(msg)

    named = _dedup(_named(packages, path))
    versions = {canonicalize_name(name): str(pkg.get("version", "")) for name, pkg in named}
    return [
        SyntheticDistribution(name, str(pkg.get("version", "")), _children(pkg.get("dependencies", ()), versions))
        for name, pkg in named
    ]


def _named(packages: list[Any], path: Path) -> list[tuple[str, dict[str, Any]]]:
    named: list[tuple[str, dict[str, Any]]] = []
    for pkg in packages:
        if not isinstance(pkg, dict) or "name" not in pkg:
            msg = f"not a valid PEP 751 lock file: {path} (a package entry is missing 'name')"
            raise FromLockError(msg)
        named.append((str(pkg["name"]), pkg))
    return named


def _dedup(named: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Merge entries that share a canonical name, preferring the first record that carries a version."""
    order: dict[NormalizedName, int] = {}
    result: list[tuple[str, dict[str, Any]]] = []
    for name, pkg in named:
        canon = canonicalize_name(name)
        if canon not in order:
            order[canon] = len(result)
            result.append((name, dict(pkg)))
        else:
            idx = order[canon]
            winner_name, winner = result[idx]
            if not winner.get("version") and pkg.get("version"):
                winner["version"] = pkg["version"]
                winner_name = name
            winner["dependencies"] = _merge_deps(
                winner.get("dependencies", []),
                pkg.get("dependencies", []),
            )
            result[idx] = (winner_name, winner)
    return result


def _merge_deps(base: list[Any], extra: list[Any]) -> list[Any]:
    """Union two dependency lists, deduplicating by canonical child name."""
    seen: set[NormalizedName] = set()
    merged: list[Any] = []
    for dep in (*base, *extra):
        canon = canonicalize_name(str(dep["name"]))
        if canon not in seen:
            seen.add(canon)
            merged.append(dep)
    return merged


def _children(dependencies: Any, versions: dict[NormalizedName, str]) -> tuple[str, ...]:
    # A leaf omits the ``dependencies`` key entirely, so the default () yields no edges. Each edge carries only the
    # child name; pin it from the looked-up version when known, else emit a bare requirement.
    children: list[str] = []
    for dep in dependencies:
        child = str(dep["name"])
        children.append(f"{child}=={version}" if (version := versions.get(canonicalize_name(child))) else child)
    return tuple(children)


__all__ = [
    "FromLockError",
    "load_lock",
]
