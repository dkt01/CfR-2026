"""Parsers for the Arduino's diagnostic serial lines.

Both are emitted by the firmware and skipped by arduino_bridge_node, which
writes them straight into the run directory's rx trace.  Field orders here must
match the snprintf calls in src/arduino_rcm/arduino_rcm.ino; the tests pin them
against sample lines so a firmware change that reorders a field fails the build
rather than silently mislabelling a column six months later.

Lines may carry a host timestamp prefix, which arduino_bridge_node adds when
trace_timestamps is set (characterize.launch.py always sets it).  The Arduino
stamps its own lines with millis() since ITS boot, which says nothing about when
they arrived; the host stamp is what makes them joinable with telemetry.csv.
"""

import math

__all__ = [
    "DEBUG_FIELDS",
    "TRACE_FIELDS",
    "parse_line",
    "parse_trace_file",
    "ParsedLine",
]

# "D,%lu,%u,%u,%u,%u,%d,%u,%u,%u,%u,%u,%u,%s,%u,%u,%u,%u\n"
DEBUG_FIELDS = [
    ("arduino_ms", int),
    ("valid_frames", int),
    ("invalid_frames", int),
    ("auto_ready", int),
    ("cmd_steering", int),
    ("target_rpm", int),
    ("mode", int),
    ("steering_us", int),
    ("throttle_us", int),
    ("rpm", int),
    ("battery_mv", int),
    ("rejected_len", int),
    ("rejected_hex", str),
    ("dither_duty", int),
    ("gains_seq", int),
    ("merged_pulses", int),
    ("rejected_edges", int),
]

# "T,%lu,%d,%u,%d,%d,%d,%d,%d,%u,%u,%u\n"
# The five controller terms are in TENTHS of a microsecond, scaled back here so
# every consumer works in microseconds and nobody has to remember.
TRACE_FIELDS = [
    ("arduino_ms", int),
    ("tracked_rpm", int),
    ("measured_rpm", int),
    ("feedforward_us", lambda raw: int(raw) / 10.0),
    ("proportional_us", lambda raw: int(raw) / 10.0),
    ("integral_us", lambda raw: int(raw) / 10.0),
    ("derivative_us", lambda raw: int(raw) / 10.0),
    ("output_us", lambda raw: int(raw) / 10.0),
    ("throttle_us", int),
    ("braking", int),
    ("dither_duty", int),
]


class ParsedLine(dict):
    """One parsed diagnostic line.  `kind` is 'D' or 'T'."""

    @property
    def kind(self):
        return self["kind"]


def _split_host_stamp(line):
    """Peel off a host timestamp prefix if one is present.

    The prefix is `<seconds> ` before the tag, so a line is timestamped exactly
    when its first token parses as a float and the next one starts with a tag.
    """
    stripped = line.strip()
    if not stripped:
        return None, ""
    head, _, tail = stripped.partition(" ")
    if tail[:2] in ("D,", "T,"):
        try:
            return float(head), tail
        except ValueError:
            return None, stripped
    return None, stripped


def parse_line(line):
    """Parse one trace line, or return None if it is not a D,/T, line.

    Malformed lines return None rather than raising: serial traces legitimately
    contain truncated lines where a frame was cut off, and one bad line must not
    cost the rest of a run's data.
    """
    host_time, body = _split_host_stamp(line)
    if body[:2] == "D,":
        fields = DEBUG_FIELDS
    elif body[:2] == "T,":
        fields = TRACE_FIELDS
    else:
        return None

    parts = body[2:].split(",")
    # The firmware's status frame ends with a trailing comma, and the debug line
    # can too; drop one empty tail rather than treating it as a missing field.
    if parts and parts[-1] == "":
        parts.pop()
    if len(parts) != len(fields):
        return None

    parsed = ParsedLine(kind=body[0], host_time=host_time)
    try:
        for (name, convert), raw in zip(fields, parts):
            parsed[name] = convert(raw)
    except (ValueError, TypeError):
        return None
    return parsed


def parse_trace_file(path):
    """Parse a whole rx trace, returning (debug_lines, trace_lines, skipped)."""
    debug, trace, skipped = [], [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            parsed = parse_line(line)
            if parsed is None:
                skipped += 1
            elif parsed.kind == "D":
                debug.append(parsed)
            else:
                trace.append(parsed)
    return debug, trace, skipped


def align_arduino_clock(lines):
    """Best-fit seconds-per-millisecond and offset from Arduino ms to host time.

    The Arduino's millis() and the host clock drift apart, so a run of any
    length needs the relationship fitted rather than assuming one stamp lines
    both up.  Returns None when too few lines carry a host stamp.
    """
    from .linalg import fit_line

    stamped = [
        (line["arduino_ms"], line["host_time"])
        for line in lines
        if line.get("host_time") is not None
    ]
    if len(stamped) < 10:
        return None
    slope, intercept, r_squared = fit_line(
        [ms for ms, _ in stamped], [host for _, host in stamped]
    )
    if not math.isfinite(slope) or slope <= 0.0:
        return None
    return {
        "scale": slope,
        "offset": intercept,
        "r_squared": r_squared,
        "samples": len(stamped),
    }
