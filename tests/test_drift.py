from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pipdeptree._drift import DriftEntry, DriftReport, compute_drift
from pipdeptree._models import PackageDAG
from pipdeptree._render.drift import render_drift_json, render_drift_text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from unittest.mock import Mock

    from pytest_mock import MockerFixture

    from tests.our_types import MockGraph


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dag(mock_pkgs: Callable[[MockGraph], Iterator[Mock]], graph: MockGraph) -> PackageDAG:
    return PackageDAG.from_pkgs(list(mock_pkgs(graph)))


def _write_lock(tmp_path: Path, body: str) -> Path:
    lock = tmp_path / "pylock.toml"
    lock.write_text(body, encoding="utf-8")
    return lock


# ---------------------------------------------------------------------------
# compute_drift unit tests
# ---------------------------------------------------------------------------


class TestComputeDrift:
    def test_identical_trees(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("b", "1.2"): [],
        }
        env = _dag(mock_pkgs, graph)
        ref = _dag(mock_pkgs, graph)
        report = compute_drift(env, ref)
        assert not report.has_drift
        assert report.missing == []
        assert report.extra == []
        assert report.version_mismatch == []

    def test_missing_packages(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {("a", "1.0"): []}
        ref_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("b", "1.2"): [],
        }
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert len(report.missing) == 1
        assert report.missing[0].key == "b"
        assert report.missing[0].ref_version == "1.2"
        assert report.missing[0].env_version is None
        assert report.missing[0].kind == "missing"

    def test_extra_packages(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {
            ("a", "1.0"): [],
            ("debug", "0.5"): [],
        }
        ref_graph: MockGraph = {("a", "1.0"): []}
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert len(report.extra) == 1
        assert report.extra[0].key == "debug"
        assert report.extra[0].env_version == "0.5"
        assert report.extra[0].ref_version is None

    def test_version_mismatch(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("b", "1.2"): [],
        }
        ref_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "2.0")])],
            ("b", "2.0"): [],
        }
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert len(report.version_mismatch) == 1
        m = report.version_mismatch[0]
        assert m.key == "b"
        assert m.env_version == "1.2"
        assert m.ref_version == "2.0"
        assert m.kind == "version_mismatch"

    def test_mixed_scenario(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("b", "1.2"): [],
            ("extra-pkg", "0.1"): [],
        }
        ref_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "2.0")]), ("c", [(">=", "3.0")])],
            ("b", "2.0"): [],
            ("c", "3.1"): [],
        }
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert len(report.missing) == 1
        assert report.missing[0].key == "c"
        assert len(report.extra) == 1
        assert report.extra[0].key == "extra-pkg"
        assert len(report.version_mismatch) == 1
        assert report.version_mismatch[0].key == "b"

    def test_dependents_traced(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {("a", "1.0"): [], ("c", "2.0"): []}
        ref_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("c", "2.0"): [("b", [(">=", "1.0")])],
            ("b", "1.5"): [],
        }
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert len(report.missing) == 1
        assert report.missing[0].key == "b"
        assert sorted(report.missing[0].dependents) == ["a", "c"]

    def test_version_mismatch_dependents(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("b", "1.0"): [],
        }
        ref_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "2.0")])],
            ("b", "2.0"): [],
        }
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert report.version_mismatch[0].dependents == ("a",)

    def test_sorted_output(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {("z", "1.0"): [], ("m", "1.0"): [], ("a", "1.0"): []}
        ref_graph: MockGraph = {("z", "2.0"): [], ("m", "2.0"): [], ("a", "2.0"): []}
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        keys = [e.key for e in report.version_mismatch]
        assert keys == sorted(keys)

    def test_prerelease_version_diff(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {("pkg", "1.0rc1"): []}
        ref_graph: MockGraph = {("pkg", "1.0"): []}
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert len(report.version_mismatch) == 1
        assert report.version_mismatch[0].env_version == "1.0rc1"
        assert report.version_mismatch[0].ref_version == "1.0"

    def test_empty_trees(self, mock_pkgs: Callable[[MockGraph], Iterator[Mock]]) -> None:
        env_graph: MockGraph = {}
        ref_graph: MockGraph = {}
        report = compute_drift(_dag(mock_pkgs, env_graph), _dag(mock_pkgs, ref_graph))
        assert not report.has_drift


# ---------------------------------------------------------------------------
# DriftReport.as_dict
# ---------------------------------------------------------------------------


class TestDriftReportAsDict:
    def test_empty(self) -> None:
        report = DriftReport()
        d = report.as_dict()
        assert d == {"missing": [], "extra": [], "version_mismatch": []}

    def test_populated(self) -> None:
        report = DriftReport(
            missing=[DriftEntry(key="b", package_name="B", kind="missing", ref_version="1.0", dependents=("a",))],
            extra=[DriftEntry(key="x", package_name="X", kind="extra", env_version="0.1")],
            version_mismatch=[
                DriftEntry(key="c", package_name="C", kind="version_mismatch", env_version="1.0", ref_version="2.0")
            ],
        )
        d = report.as_dict()
        assert d["missing"][0]["key"] == "b"
        assert d["missing"][0]["ref_version"] == "1.0"
        assert d["missing"][0]["dependents"] == ["a"]
        assert "env_version" not in d["missing"][0]
        assert d["extra"][0]["env_version"] == "0.1"
        assert "ref_version" not in d["extra"][0]
        assert d["version_mismatch"][0]["env_version"] == "1.0"
        assert d["version_mismatch"][0]["ref_version"] == "2.0"


# ---------------------------------------------------------------------------
# render tests
# ---------------------------------------------------------------------------


class TestRenderDriftText:
    def test_no_drift(self, capsys: pytest.CaptureFixture[str]) -> None:
        render_drift_text(DriftReport())
        assert capsys.readouterr().out == "No drift detected.\n"

    def test_all_categories(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = DriftReport(
            missing=[DriftEntry(key="b", package_name="B", kind="missing", ref_version="1.0", dependents=("A",))],
            extra=[DriftEntry(key="x", package_name="X", kind="extra", env_version="0.1")],
            version_mismatch=[
                DriftEntry(
                    key="c",
                    package_name="C",
                    kind="version_mismatch",
                    env_version="1.0",
                    ref_version="2.0",
                    dependents=("A", "B"),
                )
            ],
        )
        render_drift_text(report)
        out = capsys.readouterr().out
        assert "Drift: 1 missing, 1 extra, 1 version mismatch\n" in out
        assert "B==1.0 (required by: A)" in out
        assert "X==0.1" in out
        assert "C: env=1.0, ref=2.0 (required by: A, B)" in out

    def test_missing_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = DriftReport(
            missing=[
                DriftEntry(key="a", package_name="a", kind="missing", ref_version="1.0"),
                DriftEntry(key="b", package_name="b", kind="missing", ref_version="2.0"),
            ],
        )
        render_drift_text(report)
        out = capsys.readouterr().out
        assert "2 missing" in out
        assert "Extra" not in out
        assert "Version mismatch" not in out

    def test_plural_mismatches(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = DriftReport(
            version_mismatch=[
                DriftEntry(key="a", package_name="a", kind="version_mismatch", env_version="1", ref_version="2"),
                DriftEntry(key="b", package_name="b", kind="version_mismatch", env_version="3", ref_version="4"),
            ],
        )
        render_drift_text(report)
        out = capsys.readouterr().out
        assert "2 version mismatches" in out


class TestRenderDriftJson:
    def test_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = DriftReport(
            missing=[DriftEntry(key="b", package_name="B", kind="missing", ref_version="1.0", dependents=("A",))],
        )
        render_drift_json(report)
        data = json.loads(capsys.readouterr().out)
        assert "missing" in data
        assert "extra" in data
        assert "version_mismatch" in data
        assert data["missing"][0]["key"] == "b"

    def test_empty_report_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        render_drift_json(DriftReport())
        data = json.loads(capsys.readouterr().out)
        assert data == {"missing": [], "extra": [], "version_mismatch": []}


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


_LOCK_ENV_MATCH = """\
lock-version = "1.0"
[[packages]]
name = "a"
version = "1.0"
[[packages]]
name = "b"
version = "2.0"
[[packages.dependencies]]
name = "a"
"""

_LOCK_WITH_DRIFT = """\
lock-version = "1.0"
[[packages]]
name = "a"
version = "1.0"
[[packages.dependencies]]
name = "b"
[[packages]]
name = "b"
version = "3.0"
[[packages]]
name = "c"
version = "1.5"
"""


class TestDriftCli:
    def test_drift_from_lock_no_drift(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        mock_pkgs: Callable[[MockGraph], Iterator[Mock]],
    ) -> None:
        from pipdeptree.__main__ import main

        env_graph: MockGraph = {
            ("a", "1.0"): [],
            ("b", "2.0"): [("a", [(">=", "1.0")])],
        }
        dists = list(mock_pkgs(env_graph))
        mocker.patch("pipdeptree.__main__.get_installed_distributions", return_value=dists)
        mocker.patch("pipdeptree.__main__._resolve_python", return_value="python")
        lock = _write_lock(tmp_path, _LOCK_ENV_MATCH)
        result = main(["drift", "from-lock", str(lock)])
        assert result == 0

    def test_drift_from_lock_with_drift(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        mock_pkgs: Callable[[MockGraph], Iterator[Mock]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from pipdeptree.__main__ import main

        env_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("b", "1.0"): [],
            ("extra-pkg", "0.1"): [],
        }
        dists = list(mock_pkgs(env_graph))
        mocker.patch("pipdeptree.__main__.get_installed_distributions", return_value=dists)
        mocker.patch("pipdeptree.__main__._resolve_python", return_value="python")
        lock = _write_lock(tmp_path, _LOCK_WITH_DRIFT)
        result = main(["drift", "from-lock", str(lock)])
        assert result == 1
        out = capsys.readouterr().out
        assert "missing" in out.lower() or "Missing" in out
        assert "extra" in out.lower() or "Extra" in out

    def test_drift_from_lock_json_output(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        mock_pkgs: Callable[[MockGraph], Iterator[Mock]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from pipdeptree.__main__ import main

        env_graph: MockGraph = {
            ("a", "1.0"): [("b", [(">=", "1.0")])],
            ("b", "1.0"): [],
        }
        dists = list(mock_pkgs(env_graph))
        mocker.patch("pipdeptree.__main__.get_installed_distributions", return_value=dists)
        mocker.patch("pipdeptree.__main__._resolve_python", return_value="python")
        lock = _write_lock(tmp_path, _LOCK_WITH_DRIFT)
        result = main(["drift", "from-lock", str(lock), "-o", "json"])
        assert result == 1
        data = json.loads(capsys.readouterr().out)
        assert "missing" in data
        assert "extra" in data
        assert "version_mismatch" in data

    def test_drift_requires_source(self) -> None:
        from pipdeptree._cli import get_options

        with pytest.raises(SystemExit):
            get_options(["drift"])

    def test_drift_from_lock_bad_file(self, tmp_path: Path, mocker: MockerFixture) -> None:
        from pipdeptree.__main__ import main

        mocker.patch("pipdeptree.__main__.get_installed_distributions", return_value=[])
        mocker.patch("pipdeptree.__main__._resolve_python", return_value="python")
        result = main(["drift", "from-lock", str(tmp_path / "nonexistent.toml")])
        assert result == 1
