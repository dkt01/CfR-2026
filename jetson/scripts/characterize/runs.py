"""Loading a run directory.

A run directory is what record-side tooling produces and what everything here
consumes:

    <run>/telemetry.csv    the primary artifact: one row per 50 Hz control tick
    <run>/arduino_rx.log   host-timestamped D, and T, lines from the firmware
    <run>/arduino_tx.log   host-timestamped frames written to the firmware
    <run>/metadata.yaml    profile, gains, git SHA, battery, result
    <run>/runner.log       what the runner said while it ran
    <run>/bag/             rosbag2, raw backup

telemetry.csv is deliberately the primary one rather than the bag: it needs no
rosbag2_py, opens in any spreadsheet, and survives being emailed around.
"""

import csv
import os

__all__ = ["Run", "Segment", "load"]

_NUMERIC = {
    "t_ros",
    "t_elapsed",
    "cmd_steering",
    "cmd_velocity",
    "battery_volts",
    "wheel_rpm",
    "speed",
    "target_speed",
    "odom_x",
    "odom_y",
    "odom_yaw",
    "odom_vx",
    "odom_wz",
    "dist_along",
    "dist_total",
}
_INTEGER = {
    "step_index",
    "auto_ready",
    "mode",
    "link_ok",
    "estop",
    "gains_applied",
    "battery_level",
    "rpm",
    "throttle_us",
    "odom_valid",
}


def _coerce(name, raw):
    if raw == "" or raw is None:
        return None
    try:
        if name in _INTEGER:
            return int(raw)
        if name in _NUMERIC:
            return float(raw)
    except ValueError:
        return None
    return raw


class Segment:
    """One step of a profile, as actually executed.

    `hold` rows only - the gains_wait window between steps is excluded, because
    during it the car is still running the PREVIOUS step's command while the new
    gains propagate.  Including those rows would attribute one step's motion to
    the next one's label.
    """

    def __init__(self, index, label, rows):
        self.index = index
        self.label = label
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"<Segment {self.index} {self.label!r} rows={len(self.rows)}>"

    def column(self, name):
        return [row[name] for row in self.rows if row.get(name) is not None]

    def pair(self, x_name, y_name):
        """Two columns, restricted to rows where both are present."""
        xs, ys = [], []
        for row in self.rows:
            if row.get(x_name) is not None and row.get(y_name) is not None:
                xs.append(row[x_name])
                ys.append(row[y_name])
        return xs, ys

    @property
    def duration(self):
        times = self.column("t_ros")
        return times[-1] - times[0] if len(times) > 1 else 0.0


class Run:
    def __init__(self, path, rows, metadata, debug_lines, trace_lines, skipped_lines):
        self.path = path
        self.rows = rows
        self.metadata = metadata
        self.debug_lines = debug_lines
        self.trace_lines = trace_lines
        self.skipped_lines = skipped_lines

    @property
    def profile(self):
        return self.metadata.get("profile", os.path.basename(self.path))

    @property
    def result(self):
        return self.metadata.get("result", "unknown")

    @property
    def aborted(self):
        return self.result not in ("ok",)

    def parameter(self, name, default=None):
        """A bridge parameter as recorded at run time, not as configured now."""
        params = self.metadata.get("bridge_parameters") or {}
        node = params.get("arduino_bridge") or params.get("/arduino_bridge") or {}
        values = node.get("ros__parameters", node)
        return values.get(name, default)

    def segments(self, phase="running"):
        """Executed steps, in order, excluding gains_wait rows (see Segment)."""
        found, current, key = [], [], None
        for row in self.rows:
            if row.get("phase") != phase or row.get("step_phase") != "hold":
                continue
            row_key = (row.get("step_index"), row.get("step_label"))
            if row_key != key:
                if current:
                    found.append(Segment(key[0], key[1], current))
                current, key = [], row_key
            current.append(row)
        if current:
            found.append(Segment(key[0], key[1], current))
        return found

    def segment(self, label):
        for segment in self.segments():
            if segment.label == label:
                return segment
        return None

    def matching(self, prefix):
        return [
            segment
            for segment in self.segments()
            if (segment.label or "").startswith(prefix)
        ]

    def battery_span(self):
        levels = [
            row["battery_level"]
            for row in self.rows
            if row.get("battery_level") is not None
        ]
        millivolts = [
            line["battery_mv"] for line in self.debug_lines if line.get("battery_mv")
        ]
        span = {}
        if levels:
            span["level_start"], span["level_end"] = levels[0], levels[-1]
        if millivolts:
            span["mv_start"], span["mv_end"] = millivolts[0], millivolts[-1]
            span["mv_min"] = min(millivolts)
        return span

    def tach_health(self):
        """Merged pulses and rejected edges, from the D, line's own counters.

        These are free: every run logs them, so tachometer behaviour on the
        ground never needs a dedicated test.  On the bench roughly 5% of
        revolutions were merged; a much larger figure here means loop stalls or
        electrical noise on the sensor line.
        """
        if not self.debug_lines:
            return {}
        first, last = self.debug_lines[0], self.debug_lines[-1]
        merged = last.get("merged_pulses", 0) - first.get("merged_pulses", 0)
        rejected = last.get("rejected_edges", 0) - first.get("rejected_edges", 0)
        invalid = last.get("invalid_frames", 0) - first.get("invalid_frames", 0)
        valid = last.get("valid_frames", 0) - first.get("valid_frames", 0)
        return {
            "merged_pulses": merged,
            "rejected_edges": rejected,
            "invalid_frames": invalid,
            "valid_frames": valid,
            "frame_loss_fraction": invalid / (valid + invalid)
            if (valid + invalid)
            else 0.0,
        }


def load(path):
    """Load a run directory.  Missing optional files are tolerated."""
    path = os.path.abspath(os.path.expanduser(path))
    telemetry = os.path.join(path, "telemetry.csv")
    if not os.path.isfile(telemetry):
        raise FileNotFoundError(f"{path} has no telemetry.csv - is it a run directory?")

    rows = []
    with open(telemetry, "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({name: _coerce(name, value) for name, value in raw.items()})

    metadata = {}
    metadata_path = os.path.join(path, "metadata.yaml")
    if os.path.isfile(metadata_path):
        import yaml

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle) or {}

    debug_lines, trace_lines, skipped = [], [], 0
    rx_path = os.path.join(path, "arduino_rx.log")
    if os.path.isfile(rx_path):
        from .traces import parse_trace_file

        debug_lines, trace_lines, skipped = parse_trace_file(rx_path)

    return Run(path, rows, metadata, debug_lines, trace_lines, skipped)
