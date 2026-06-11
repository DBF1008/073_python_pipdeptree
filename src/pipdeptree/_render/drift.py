from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipdeptree._drift import DriftReport


def render_drift_text(report: DriftReport) -> None:
    """Render a human-readable drift report to stdout."""
    n_missing = len(report.missing)
    n_extra = len(report.extra)
    n_mismatch = len(report.version_mismatch)

    if not report.has_drift:
        print("No drift detected.")  # noqa: T201
        return

    parts: list[str] = []
    if n_missing:
        parts.append(f"{n_missing} missing")
    if n_extra:
        parts.append(f"{n_extra} extra")
    if n_mismatch:
        parts.append(f"{n_mismatch} version mismatch{'es' if n_mismatch != 1 else ''}")
    print(f"Drift: {', '.join(parts)}")  # noqa: T201

    if report.missing:
        print()  # noqa: T201
        print("Missing (in reference but not installed):")  # noqa: T201
        for e in report.missing:
            dep_info = f" (required by: {', '.join(e.dependents)})" if e.dependents else ""
            print(f"  {e.package_name}=={e.ref_version}{dep_info}")  # noqa: T201

    if report.extra:
        print()  # noqa: T201
        print("Extra (installed but not in reference):")  # noqa: T201
        for e in report.extra:
            print(f"  {e.package_name}=={e.env_version}")  # noqa: T201

    if report.version_mismatch:
        print()  # noqa: T201
        print("Version mismatch:")  # noqa: T201
        for e in report.version_mismatch:
            dep_info = f" (required by: {', '.join(e.dependents)})" if e.dependents else ""
            print(f"  {e.package_name}: env={e.env_version}, ref={e.ref_version}{dep_info}")  # noqa: T201


def render_drift_json(report: DriftReport) -> None:
    """Render a machine-readable JSON drift report to stdout."""
    json.dump(report.as_dict(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


__all__ = [
    "render_drift_json",
    "render_drift_text",
]
