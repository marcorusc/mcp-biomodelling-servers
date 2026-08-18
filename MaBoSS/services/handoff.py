"""Portable MaBoSS handoff metadata helpers."""

import math
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version


def maboss_package_version() -> str:
    """Return the installed pyMaBoSS distribution version for provenance."""
    try:
        return package_version("maboss")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot export a handoff because the installed `maboss` package "
            "version is unavailable."
        ) from exc


def handoff_parameters(
    parameters,
) -> dict[str, bool | int | float | str | None]:
    """Return exact portable scalar parameters for a handoff manifest."""
    normalized = {}
    for raw_name, raw_value in parameters.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(
                "MaBoSS contains an empty parameter name that cannot be exported."
            )
        if name in normalized:
            raise ValueError(
                f"MaBoSS parameter names collapse to duplicate key {name!r}."
            )

        value = raw_value
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if value is None or isinstance(value, (bool, int, str)):
            normalized[name] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"MaBoSS parameter {name!r} must be finite for handoff."
                )
            normalized[name] = value
        else:
            raise ValueError(
                f"MaBoSS parameter {name!r} has unsupported non-scalar type "
                f"{type(value).__name__!r}."
            )
    return normalized
