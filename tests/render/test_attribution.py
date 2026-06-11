from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from pipdeptree._models.dag import PackageDAG
from pipdeptree._render.attribution import render_attribution
from pipdeptree._synthetic_dist import SyntheticDistribution

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from importlib.metadata import Distribution
    from unittest.mock import Mock

    from pytest_mock import MockerFixture

    from tests.our_types import MockGraph


def _attribution_json(dag: PackageDAG, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    render_attribution(dag, style="json")
    return json.loads(capsys.readouterr().out)


def _attribution_text(dag: PackageDAG, capsys: pytest.CaptureFixture[str]) -> str:
    render_attribution(dag)
    return capsys.readouterr().out


def test_example_dag_roots(
    example_dag: PackageDAG, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _attribution_json(example_dag, capsys)

    roots_by_name = {r["package"]: r for r in data["roots"]}
    assert set(roots_by_name) == {"a", "g"}
    assert roots_by_name["a"]["transitive_deps"] == ["b", "c", "d", "e"]
    assert roots_by_name["g"]["transitive_deps"] == ["b", "d", "e", "f"]


def test_example_dag_overlap(
    example_dag: PackageDAG, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _attribution_json(example_dag, capsys)

    deps_by_name = {td["package"]: td for td in data["transitive_deps"]}
    assert deps_by_name["b"] == {"package": "b", "version": "2.3.1", "root_count": 2, "attributed_to": ["a", "g"]}
    assert deps_by_name["c"] == {"package": "c", "version": "5.10.0", "root_count": 1, "attributed_to": ["a"]}
    assert deps_by_name["d"] == {"package": "d", "version": "2.35", "root_count": 2, "attributed_to": ["a", "g"]}
    assert deps_by_name["e"] == {"package": "e", "version": "0.12.1", "root_count": 2, "attributed_to": ["a", "g"]}
    assert deps_by_name["f"] == {"package": "f", "version": "3.1", "root_count": 1, "attributed_to": ["g"]}


def test_example_dag_summary_stats(
    example_dag: PackageDAG, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _attribution_json(example_dag, capsys)

    assert data["summary"] == {
        "total_packages": 7,
        "root_count": 2,
        "transitive_count": 5,
        "max_overlap": 2,
        "unique_to_one_root": 2,
    }


def test_single_package_no_deps(
    mock_pkgs: Callable[[MockGraph], Iterator[Mock]], capsys: pytest.CaptureFixture[str]
) -> None:
    dag = PackageDAG.from_pkgs(list(mock_pkgs({("a", "1.0.0"): []})))

    data = _attribution_json(dag, capsys)

    assert len(data["roots"]) == 1
    assert data["roots"][0] == {"package": "a", "version": "1.0.0", "transitive_deps": []}
    assert data["transitive_deps"] == []
    assert data["summary"]["root_count"] == 1
    assert data["summary"]["transitive_count"] == 0
    assert data["summary"]["max_overlap"] == 0
    assert data["summary"]["unique_to_one_root"] == 0


def test_linear_chain(
    mock_pkgs: Callable[[MockGraph], Iterator[Mock]], capsys: pytest.CaptureFixture[str]
) -> None:
    dag = PackageDAG.from_pkgs(
        list(mock_pkgs({("a", "1.0"): [("b", [])], ("b", "2.0"): [("c", [])], ("c", "3.0"): []}))
    )

    data = _attribution_json(dag, capsys)

    assert len(data["roots"]) == 1
    assert data["roots"][0]["package"] == "a"
    assert data["roots"][0]["transitive_deps"] == ["b", "c"]
    assert all(td["root_count"] == 1 for td in data["transitive_deps"])
    assert all(td["attributed_to"] == ["a"] for td in data["transitive_deps"])
    assert data["summary"]["unique_to_one_root"] == 2


def test_diamond_single_root(
    mock_pkgs: Callable[[MockGraph], Iterator[Mock]], capsys: pytest.CaptureFixture[str]
) -> None:
    dag = PackageDAG.from_pkgs(
        list(
            mock_pkgs({
                ("a", "1.0"): [("b", []), ("c", [])],
                ("b", "1.0"): [("d", [])],
                ("c", "1.0"): [("d", [])],
                ("d", "1.0"): [],
            })
        )
    )

    data = _attribution_json(dag, capsys)

    assert len(data["roots"]) == 1
    deps_by_name = {td["package"]: td for td in data["transitive_deps"]}
    assert deps_by_name["d"]["root_count"] == 1
    assert deps_by_name["d"]["attributed_to"] == ["a"]


def test_cycle_no_roots(
    mock_pkgs: Callable[[MockGraph], Iterator[Mock]], capsys: pytest.CaptureFixture[str]
) -> None:
    dag = PackageDAG.from_pkgs(
        list(mock_pkgs({("a", "1.0"): [("b", [])], ("b", "1.0"): [("a", [])]}))
    )

    data = _attribution_json(dag, capsys)

    assert data["roots"] == []
    assert data["transitive_deps"] == []
    assert data["summary"]["root_count"] == 0
    assert data["summary"]["transitive_count"] == 0


def test_empty_tree(capsys: pytest.CaptureFixture[str]) -> None:
    data = _attribution_json(PackageDAG({}), capsys)

    assert data["roots"] == []
    assert data["transitive_deps"] == []
    assert data["summary"] == {
        "total_packages": 0,
        "root_count": 0,
        "transitive_count": 0,
        "max_overlap": 0,
        "unique_to_one_root": 0,
    }


def test_text_format_structure(
    example_dag: PackageDAG, capsys: pytest.CaptureFixture[str]
) -> None:
    text = _attribution_text(example_dag, capsys)

    assert "a (3.4.0): b, c, d, e" in text
    assert "g (6.8.3rc1): b, d, e, f" in text
    # overlap section
    assert "2 roots" in text
    assert "1 root " in text


def test_text_empty_tree(capsys: pytest.CaptureFixture[str]) -> None:
    text = _attribution_text(PackageDAG({}), capsys)

    assert "(no root packages)" in text


def test_json_schema_keys(
    example_dag: PackageDAG, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _attribution_json(example_dag, capsys)

    assert set(data.keys()) == {"roots", "transitive_deps", "summary"}
    assert set(data["summary"].keys()) == {
        "total_packages",
        "root_count",
        "transitive_count",
        "max_overlap",
        "unique_to_one_root",
    }


def test_sorted_output(
    example_dag: PackageDAG, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _attribution_json(example_dag, capsys)

    root_names = [r["package"] for r in data["roots"]]
    assert root_names == sorted(root_names)

    for root in data["roots"]:
        assert root["transitive_deps"] == sorted(root["transitive_deps"])

    dep_names = [td["package"] for td in data["transitive_deps"]]
    assert dep_names == sorted(dep_names)

    for td in data["transitive_deps"]:
        assert td["attributed_to"] == sorted(td["attributed_to"])


def test_rich_style(
    example_dag: PackageDAG, capsys: pytest.CaptureFixture[str]
) -> None:
    render_attribution(example_dag, style="rich")

    out = capsys.readouterr().out
    assert "root attribution" in out
    assert "transitive dependency overlap" in out


def test_rich_missing_import(
    mock_pkgs: Callable[[MockGraph], Iterator[Mock]], mocker: MockerFixture
) -> None:
    mocker.patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None})
    dag = PackageDAG.from_pkgs(list(mock_pkgs({("a", "1.0"): []})))

    with pytest.raises(SystemExit) as exc_info:
        render_attribution(dag, style="rich")

    assert exc_info.value.code == 1


def test_from_lock_synthetic_tree(capsys: pytest.CaptureFixture[str]) -> None:
    pkgs: list[Distribution] = [
        SyntheticDistribution("top", "1.0.0", ("mid==2.0.0",)),
        SyntheticDistribution("mid", "2.0.0", ("leaf==3.0.0",)),
        SyntheticDistribution("leaf", "3.0.0", ()),
    ]

    data = _attribution_json(PackageDAG.from_pkgs(pkgs), capsys)

    assert len(data["roots"]) == 1
    assert data["roots"][0]["package"] == "top"
    assert data["roots"][0]["transitive_deps"] == ["leaf", "mid"]
    assert data["summary"]["transitive_count"] == 2


def test_multiple_roots_shared_dep(
    mock_pkgs: Callable[[MockGraph], Iterator[Mock]], capsys: pytest.CaptureFixture[str]
) -> None:
    dag = PackageDAG.from_pkgs(
        list(
            mock_pkgs({
                ("x", "1.0"): [("shared", [])],
                ("y", "1.0"): [("shared", [])],
                ("shared", "1.0"): [],
            })
        )
    )

    data = _attribution_json(dag, capsys)

    assert len(data["roots"]) == 2
    assert data["transitive_deps"][0]["package"] == "shared"
    assert data["transitive_deps"][0]["root_count"] == 2
    assert data["transitive_deps"][0]["attributed_to"] == ["x", "y"]
    assert data["summary"]["max_overlap"] == 2
    assert data["summary"]["unique_to_one_root"] == 0
