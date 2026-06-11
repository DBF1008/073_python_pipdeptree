"""
Render a dependency attribution summary.

For each root package (one not required by any other package in the tree), compute the full transitive
closure of its dependencies.  Then, for each transitive dependency, list which roots pull it in and how
many -- surfacing shared/duplicated transitive dependencies across independent top-level packages.

The same data drives three presentation styles -- aligned ``text``, a ``rich`` table pair, and ``json`` --
all built from one shared collection pass.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipdeptree._models import PackageDAG


@dataclass
class _RootAttribution:
    package: str
    version: str
    transitive_deps: list[str]


@dataclass
class _TransitiveDep:
    package: str
    version: str
    root_count: int
    attributed_to: list[str]


@dataclass
class _Attribution:
    total_packages: int
    root_count: int
    transitive_count: int
    max_overlap: int
    unique_to_one_root: int
    roots: list[_RootAttribution] = field(default_factory=list)
    transitive_deps: list[_TransitiveDep] = field(default_factory=list)


def render_attribution(tree: PackageDAG, *, style: str = "text") -> None:
    """
    Print a dependency attribution summary.

    :param tree: the package tree
    :param style: presentation style -- ``"text"`` (aligned), ``"rich"`` (tables), or ``"json"``
    """
    attribution = _collect(tree)
    if style == "json":
        print(_as_json(attribution))  # noqa: T201
    elif style == "rich":
        _as_rich(attribution)
    else:
        print(_as_text(attribution))  # noqa: T201


def _collect(tree: PackageDAG) -> _Attribution:
    child_keys = {str(r.key) for r in chain.from_iterable(tree.values())}
    root_pkgs = sorted((p for p in tree if p.key not in child_keys), key=lambda p: p.key)
    version_map: dict[str, str] = {p.key: p.version for p in tree}

    root_closures: dict[str, set[str]] = {}
    for root in root_pkgs:
        closure: set[str] = set()
        stack = [c.key for c in tree.get_children(root.key)]
        while stack:
            dep_key = stack.pop()
            if dep_key not in closure:
                closure.add(dep_key)
                stack.extend(c.key for c in tree.get_children(dep_key))
        root_closures[root.key] = closure

    roots = [
        _RootAttribution(
            package=root.key,
            version=version_map.get(root.key, "?"),
            transitive_deps=sorted(root_closures[root.key]),
        )
        for root in root_pkgs
    ]

    dep_to_roots: dict[str, list[str]] = defaultdict(list)
    for root_key, closure in sorted(root_closures.items()):
        for dep_key in closure:
            dep_to_roots[dep_key].append(root_key)

    transitive_deps = [
        _TransitiveDep(
            package=dep_key,
            version=version_map.get(dep_key, "?"),
            root_count=len(root_keys),
            attributed_to=root_keys,
        )
        for dep_key, root_keys in sorted(dep_to_roots.items())
    ]

    max_overlap = max((td.root_count for td in transitive_deps), default=0)
    unique_count = sum(1 for td in transitive_deps if td.root_count == 1)

    return _Attribution(
        total_packages=len(tree),
        root_count=len(root_pkgs),
        transitive_count=len(transitive_deps),
        max_overlap=max_overlap,
        unique_to_one_root=unique_count,
        roots=roots,
        transitive_deps=transitive_deps,
    )


def _as_text(attribution: _Attribution) -> str:
    lines: list[str] = []

    for root in attribution.roots:
        deps = ", ".join(root.transitive_deps) if root.transitive_deps else "(none)"
        lines.append(f"{root.package} ({root.version}): {deps}")

    if not attribution.roots:
        lines.append("(no root packages)")

    if attribution.transitive_deps:
        lines.append("")
        pkg_width = max(len(td.package) for td in attribution.transitive_deps)
        count_width = max(len(str(td.root_count)) for td in attribution.transitive_deps)
        for td in attribution.transitive_deps:
            label = "root" if td.root_count == 1 else "roots"
            attributed = ", ".join(td.attributed_to)
            lines.append(f"{td.package:<{pkg_width}}  {td.root_count:>{count_width}} {label:<5}  {attributed}")

    return "\n".join(lines)


def _as_rich(attribution: _Attribution) -> None:
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
    except ImportError as exc:
        print(  # noqa: T201
            "rich is not available, but necessary for the output option. Please install it.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    roots_table = Table(title="root attribution", show_header=True, title_style="bold")
    roots_table.add_column("root", style="bold cyan", no_wrap=True)
    roots_table.add_column("version")
    roots_table.add_column("transitive deps", justify="right")
    roots_table.add_column("dependencies")
    for root in attribution.roots:
        roots_table.add_row(
            root.package,
            root.version,
            str(len(root.transitive_deps)),
            ", ".join(root.transitive_deps) or "(none)",
        )

    overlap_table = Table(title="transitive dependency overlap", show_header=True, title_style="bold")
    overlap_table.add_column("package", style="bold cyan", no_wrap=True)
    overlap_table.add_column("version")
    overlap_table.add_column("roots", justify="right")
    overlap_table.add_column("attributed to")
    for td in attribution.transitive_deps:
        overlap_table.add_row(td.package, td.version, str(td.root_count), ", ".join(td.attributed_to))

    console = Console()
    console.print(roots_table)
    console.print(overlap_table)


def _as_json(attribution: _Attribution) -> str:
    data: dict[str, object] = {
        "roots": [
            {
                "package": root.package,
                "version": root.version,
                "transitive_deps": root.transitive_deps,
            }
            for root in attribution.roots
        ],
        "transitive_deps": [
            {
                "package": td.package,
                "version": td.version,
                "root_count": td.root_count,
                "attributed_to": td.attributed_to,
            }
            for td in attribution.transitive_deps
        ],
        "summary": {
            "total_packages": attribution.total_packages,
            "root_count": attribution.root_count,
            "transitive_count": attribution.transitive_count,
            "max_overlap": attribution.max_overlap,
            "unique_to_one_root": attribution.unique_to_one_root,
        },
    }
    return json.dumps(data, indent=2)


__all__ = [
    "render_attribution",
]
