from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
from pathlib import Path
from platform import python_implementation
from typing import TYPE_CHECKING

import pytest
import virtualenv

from pipdeptree.__main__ import main
from pipdeptree._warning import get_warning_printer

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture(scope="session")
def expected_venv_pkgs() -> frozenset[str]:
    implementation = python_implementation()
    if implementation == "CPython":  # pragma: cpython cover
        expected = {"pip", "setuptools"}
    elif implementation == "PyPy":  # pragma: pypy cover
        expected = {"cffi", "greenlet", "pip", "hpy", "setuptools"}
    else:  # pragma: no cover
        raise ValueError(implementation)
    if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
        expected -= {"setuptools"}

    return frozenset(expected)


@pytest.mark.parametrize("args_joined", [True, False])
def test_custom_interpreter(
    tmp_path: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    args_joined: bool,
    expected_venv_pkgs: frozenset[str],
) -> None:
    # Delete $PYTHONPATH so that it cannot be passed to the custom interpreter process (since we don't know what
    # distribution metadata to expect when it's used).
    monkeypatch.delenv("PYTHONPATH", False)

    monkeypatch.chdir(tmp_path)
    result = virtualenv.cli_run([str(tmp_path / "venv"), "--activators", ""])
    py = str(result.creator.exe.relative_to(tmp_path))
    cmd = ["", f"--python={result.creator.exe}"] if args_joined else ["", "--python", py]
    cmd += ["--all", "--depth", "0"]
    mocker.patch("pipdeptree._discovery.sys.argv", cmd)
    main()
    out, _ = capfd.readouterr()
    found = {i.split("==")[0] for i in out.splitlines()}

    assert expected_venv_pkgs == found, out


def test_custom_interpreter_with_local_only(
    tmp_path: Path,
    mocker: MockerFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    venv_path = str(tmp_path / "venv")
    result = virtualenv.cli_run([venv_path, "--system-site-packages", "--activators", ""])

    cmd = ["", f"--python={result.creator.exe}", "--local-only"]
    mocker.patch("pipdeptree._discovery.sys.prefix", venv_path)
    mocker.patch("pipdeptree._discovery.sys.argv", cmd)
    main()
    out, _ = capfd.readouterr()
    found = {i.split("==")[0] for i in out.splitlines()}
    expected = {"pip", "setuptools"}
    if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
        expected -= {"setuptools"}
    assert expected == found, out


def test_custom_interpreter_with_user_only(
    tmp_path: Path, mocker: MockerFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    # ensures there is no output when --user-only and --python are passed

    venv_path = str(tmp_path / "venv")
    result = virtualenv.cli_run([venv_path, "--activators", ""])

    cmd = ["", f"--python={result.creator.exe}", "--user-only"]
    mocker.patch("pipdeptree.__main__.sys.argv", cmd)
    main()
    out, err = capfd.readouterr()
    assert not err

    # Here we expect 1 element because print() adds a newline.
    found = out.splitlines()
    assert len(found) == 1
    assert not found[0]


def test_custom_interpreter_with_user_only_and_system_site_pkgs_enabled(
    tmp_path: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # ensures that we provide user site metadata when --user-only and --python are passed and the custom interpreter has
    # system site packages enabled

    venv_path = str(tmp_path / "venv")
    result = virtualenv.cli_run([venv_path, "--activators", ""])
    py = str(result.creator.exe)

    # Use PYTHONUSERBASE to control the target interpreter's user site directory so that
    # site.getusersitepackages() inside the target subprocess returns a path under tmp_path.
    userbase = tmp_path / "userbase"
    env = {**os.environ, "PYTHONUSERBASE": str(userbase)}
    target_user_site = subprocess.run(
        [py, "-c", "import site; print(site.getusersitepackages())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Create a fake dist inside the target interpreter's user site.
    fake_dist_path = Path(target_user_site) / "bar-2.4.5.dist-info"
    fake_dist_path.mkdir(parents=True)
    (fake_dist_path / "METADATA").write_text("Metadata-Version: 2.3\nName: bar\nVersion: 2.4.5\n")

    # PYTHONUSERBASE controls site.getusersitepackages(); PYTHONPATH ensures the directory appears in sys.path.
    monkeypatch.setenv("PYTHONUSERBASE", str(userbase))
    monkeypatch.setenv("PYTHONPATH", target_user_site)

    cmd = ["", f"--python={py}", "--user-only"]
    mocker.patch("pipdeptree.__main__.sys.argv", cmd)
    main()

    out, err = capfd.readouterr()
    assert not err
    found = {i.split("==")[0] for i in out.splitlines()}
    expected = {"bar"}

    assert expected == found


def test_custom_interpreter_ensure_pythonpath_envar_is_honored(
    tmp_path: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    expected_venv_pkgs: frozenset[str],
) -> None:
    # ensures that we honor $PYTHONPATH when passing it to the custom interpreter process
    venv_path = str(tmp_path / "venv")
    result = virtualenv.cli_run([venv_path, "--activators", ""])

    another_path = tmp_path / "another-path"
    fake_dist = another_path / "foo-1.2.3.dist-info"
    fake_dist.mkdir(parents=True)
    fake_metadata = fake_dist / "METADATA"
    with fake_metadata.open("w") as f:
        f.write("Metadata-Version: 2.3\nName: foo\nVersion: 1.2.3\n")
    cmd = ["", f"--python={result.creator.exe}", "--all", "--depth", "0"]
    mocker.patch("pipdeptree._discovery.sys.argv", cmd)
    monkeypatch.setenv("PYTHONPATH", str(another_path))
    main()
    out, _ = capfd.readouterr()
    found = {i.split("==")[0] for i in out.splitlines()}
    assert {*expected_venv_pkgs, "foo"} == found, out


def test_custom_interpreter_user_only_no_spurious_warnings(
    tmp_path: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """--python + --user-only must not emit duplicate/invalid metadata warnings from other site dirs."""
    get_warning_printer()._has_warned = False  # noqa: SLF001

    venv_path = str(tmp_path / "venv")
    result = virtualenv.cli_run([venv_path, "--activators", ""])
    py = str(result.creator.exe)

    monkeypatch.delenv("PYTHONPATH", raising=False)

    cmd = ["", f"--python={py}", "--user-only", "-w", "fail"]
    mocker.patch("pipdeptree.__main__.sys.argv", cmd)
    ret = main()

    _, err = capfd.readouterr()
    assert "Warning" not in err
    assert ret == 0


def test_custom_interpreter_local_only_no_spurious_warnings(
    tmp_path: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """--python + --local-only must not emit duplicate/invalid metadata warnings from global site dirs."""
    get_warning_printer()._has_warned = False  # noqa: SLF001

    venv_path = str(tmp_path / "venv")
    result = virtualenv.cli_run([venv_path, "--system-site-packages", "--activators", ""])
    py = str(result.creator.exe)

    monkeypatch.delenv("PYTHONPATH", raising=False)

    cmd = ["", f"--python={py}", "--local-only", "-w", "fail"]
    mocker.patch("pipdeptree.__main__.sys.argv", cmd)
    ret = main()

    _, err = capfd.readouterr()
    assert "Warning" not in err
    assert ret == 0
