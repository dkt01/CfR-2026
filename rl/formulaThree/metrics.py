"""Shared clean-finish and checkpoint-ranking contract."""

import numpy as np


def is_clean(record):
    return bool(
        record.get("clean_finish", False)
        and record["stopped"]
        and not record["crashed"]
        and np.isfinite(record["race_time"])
    )


def selection_key(nominal, field):
    """Reliability before pace; partial progress only breaks no-finish ties."""

    def pace(summary):
        value = summary["lap"]
        return -value if np.isfinite(value) else -1e9

    return (
        nominal["finish"],
        field["finish"],
        pace(field),
        pace(nominal),
        field["distance"],
        field["clearance_p10"],
    )


def json_ready(value):
    """Failed races have no lap time: use JSON null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
