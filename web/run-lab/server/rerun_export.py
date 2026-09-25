"""One run as a Rerun recording: <run>/analysis/recording.rrd.

The Run Lab computes what a run MEANS (verdicts, sections, the plant
comparison); Rerun is where the raw material is looked at -- the 3D course
with the car, the ZED clouds, the camera, every channel against time, the
logs.  This module writes that material, already placed in the TRACK frame
and on the analysis timeline, so the viewer and the Run Lab's own playhead
show the same instant for the same number.

Entity layout (Z up, metres, track frame):

    world/course/bales        Boxes3D, static
    world/course/centerline   LineStrips3D coloured by cap zone, static
    world/path                the driven path, coloured by speed, static
    world/events              contacts / grazes / stalls / pose jumps, static
    world/car                 Transform3D over time; world/car/body, /nose
    world/cloud/frame         the ZED cloud at the playhead
    world/cloud/map           the clouds accumulated into one voxel map, static
    world/cloud/zed_map       the ZED's own spatial map (--map), static
    pose2d/pose               the ZED's raw pose in its OWN map frame, z = 0,
                              static; pose2d/jumps the steps
                              no car makes; pose2d/car the pose over time.
                              Only the "pose2d" layout shows them.
    camera                    EncodedImage frames
    metrics/<group>/<name>    Scalars; one TimeSeriesView per group
    events, rosout            TextLog

Timeline: "run", a duration in seconds -- the same t the Run Lab uses (sim
time for a simulated run).  The saved blueprint lays the views out; users can
rearrange and Rerun keeps their layout per recording.  blueprint(follow=True)
is the same layout opened on the "Follow car" view, rooted at world/car
(the Replay page's toggle).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

TIMELINE = "run"
APP_ID = "cfr_run_lab"

# Reference palette (web/run-lab/frontend/src/style.css), as RGB.
SERIES = [
    (42, 120, 214),
    (235, 104, 52),
    (27, 175, 122),
    (237, 161, 0),
    (232, 123, 164),
    (0, 131, 0),
    (74, 58, 167),
    (227, 73, 72),
]
GREY = (140, 140, 135)
STATUS = {
    "bad": (208, 59, 59),
    "warn": (250, 178, 25),
    "good": (12, 163, 12),
    "info": (140, 140, 135),
}
ZONE = {
    0: (208, 59, 59),
    1: (250, 178, 25),
    2: (12, 163, 12),
}  # hairpin, taper, straight
BLUE_RAMP = np.array(
    [
        (134, 182, 239),
        (85, 152, 231),
        (42, 120, 214),
        (28, 92, 171),
        (16, 66, 129),
        (13, 54, 107),
    ],
    dtype=float,
)

# (group, entity, column, label, colour).  Groups become TimeSeriesViews.
CHANNELS = [
    ("speed", "measured", "speed", "measured (tach)", SERIES[0]),
    ("speed", "commanded", "cmd_speed", "commanded", SERIES[1]),
    ("speed", "target", "target_speed", "Arduino target", SERIES[2]),
    ("speed", "cap", "v_cap", "rule cap", GREY),
    ("speed", "floor", "v_floor", "floor", SERIES[3]),
    ("speed", "zed", "speed_pose", "ZED ground speed", SERIES[6]),
    ("steering", "command", "cmd_steer", "on the wire", SERIES[0]),
    ("steering", "prior", "steer_ff", "centerline prior", SERIES[1]),
    ("steering", "residual", "residual", "policy residual", SERIES[2]),
    ("yaw_rate", "measured", "yaw_rate", "measured", SERIES[0]),
    ("yaw_rate", "plant", "yaw_rate_plant", "plant prediction", SERIES[1]),
    ("yaw_rate", "driver", "tel_yaw_rate", "driver's estimate", SERIES[2]),
    ("clearance", "clearance", "clearance", "body to bale", SERIES[0]),
    ("clearance", "driver", "tel_clearance", "driver's view", SERIES[2]),
    ("cte", "cte", "cte", "cross-track (m, + left)", SERIES[0]),
    ("cte", "heading", "heading_err", "heading error (rad)", SERIES[1]),
    ("accel", "longitudinal", "accel", "longitudinal", SERIES[0]),
    ("accel", "lateral", "lat_acc", "lateral (v x yaw rate)", SERIES[1]),
    ("accel", "imu_x", "imu_ax", "IMU x", SERIES[2]),
    ("accel", "imu_y", "imu_ay", "IMU y", SERIES[4]),
    ("actions", "steer", "act_steer", "steer action", SERIES[0]),
    ("actions", "throttle", "act_throttle", "throttle action", SERIES[1]),
    ("progress", "station", "station", "station (m)", SERIES[0]),
    ("progress", "lap", "lap", "lap", SERIES[1]),
    ("health", "pose_age", "pose_age", "pose age (s)", SERIES[0]),
    ("health", "link_ok", "link_ok", "Arduino link ok", SERIES[2]),
    ("health", "odom_divergence", "odom_divergence", "odom - pose (m)", SERIES[3]),
    ("battery", "pack", "battery", "pack (V, approx.)", SERIES[0]),
]
GROUP_TITLES = {
    "speed": "Speed (m/s)",
    "steering": "Steering (normalised)",
    "yaw_rate": "Yaw rate (rad/s)",
    "clearance": "Clearance to bales (m)",
    "cte": "Line",
    "accel": "Acceleration (m/s²)",
    "actions": "Policy actions",
    "progress": "Progress",
    "health": "Localisation & link",
    "battery": "Battery",
}


class RunRecording:
    """A RecordingStream writing straight to <path>, with the run's layout."""

    def __init__(self, path: Path, run_name: str):
        self.path = Path(path)
        self.rec = rr.RecordingStream(APP_ID, recording_id=run_name)
        self.rec.save(str(self.path))

    def layout(self, camera_size=None):
        """The saved blueprint.  Sent once the camera's frame size is known
        (the 2D view is pinned to it), before any bulk data is written."""
        self.rec.send_blueprint(blueprint(camera_size))

    def at(self, t):
        self.rec.set_time(TIMELINE, duration=float(t))

    def close(self):
        self.rec.flush()
        self.rec.disconnect()

    # ------------------------------------------------------------- course

    def course(self, geometry):
        bales = geometry["bales"]
        bl, bw = geometry["bale_size"]
        self.rec.log(
            "world/course/bales",
            rr.Boxes3D(
                centers=[[b["x"], b["y"], 0.18] for b in bales],
                half_sizes=[[bl / 2, bw / 2, 0.18]] * len(bales),
                rotation_axis_angles=[
                    rr.RotationAxisAngle(axis=[0, 0, 1], radians=b["yaw"])
                    for b in bales
                ],
                colors=[(200, 178, 122, 90)] * len(bales),
                fill_mode="solid",
            ),
            static=True,
        )
        cl = geometry["centerline"]
        strips, colors = [], []
        start = 0
        zones = cl["zone"]
        for k in range(1, len(zones) + 1):
            if k == len(zones) or zones[k] != zones[start]:
                end = min(k + 1, len(zones))
                strips.append(
                    [
                        [x, y, 0.01]
                        for x, y in zip(cl["x"][start:end], cl["y"][start:end])
                    ]
                )
                colors.append(ZONE[zones[start]])
                start = k
        self.rec.log(
            "world/course/centerline",
            rr.LineStrips3D(strips, colors=colors, radii=0.02),
            static=True,
        )
        s = geometry["start"]
        nx, ny = -math.sin(s["yaw"]), math.cos(s["yaw"])
        self.rec.log(
            "world/course/start_line",
            rr.LineStrips3D(
                [
                    [
                        [s["x"] - 0.46 * nx, s["y"] - 0.46 * ny, 0.02],
                        [s["x"] + 0.46 * nx, s["y"] + 0.46 * ny, 0.02],
                    ]
                ],
                colors=[(255, 255, 255)],
                radii=0.03,
            ),
            static=True,
        )

    # --------------------------------------------------------------- car

    def car(self, series, geometry, window):
        c = series["columns"]
        if "x" not in c:
            return
        t = _arr(c["t"])
        x, y, yaw = _arr(c["x"]), _arr(c["y"]), _arr(c["yaw"])
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw)
        if not ok.any():
            return
        length, width = geometry.get("car", [0.58, 0.32])
        self.rec.log(
            "world/car/body",
            rr.Boxes3D(
                half_sizes=[[length / 2, width / 2, 0.08]],
                centers=[[0, 0, 0.08]],
                colors=[SERIES[0]],
                fill_mode="solid",
            ),
            static=True,
        )
        self.rec.log(
            "world/car/nose",
            rr.Arrows3D(
                origins=[[0, 0, 0.2]],
                vectors=[[length * 0.8, 0, 0]],
                colors=[(255, 255, 255)],
                radii=0.02,
            ),
            static=True,
        )
        half = yaw[ok] / 2
        rr.send_columns(
            "world/car",
            indexes=[rr.TimeColumn(TIMELINE, duration=t[ok])],
            columns=rr.Transform3D.columns(
                translation=np.column_stack([x[ok], y[ok], np.zeros(ok.sum())]),
                quaternion=np.column_stack(
                    [np.zeros(ok.sum()), np.zeros(ok.sum()), np.sin(half), np.cos(half)]
                ),
            ),
            recording=self.rec,
        )
        # The driven path, coloured by speed, inside the run window.
        speed = np.abs(_arr(c.get("speed") or c.get("speed_pose") or [np.nan] * len(t)))
        inside = ok & (t >= window["t0"] - 0.5) & (t <= window["t1"] + 0.5)
        idx = np.flatnonzero(inside)
        segs, cols = [], []
        vmax = geometry.get("v_straight", 5.2)
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue
            segs.append([[x[a], y[a], 0.03], [x[b], y[b], 0.03]])
            cols.append(_ramp(speed[b] / vmax if np.isfinite(speed[b]) else 0.0))
        if segs:
            self.rec.log(
                "world/path",
                rr.LineStrips3D(segs, colors=cols, radii=0.025),
                static=True,
            )

    def pose2d(self, series):
        """The ZED's pose exactly as reported, in its map frame,
        flat at z = 0: the localisation, before any track fit.  A pose jump
        is drawn red; the path is broken there and at gaps, so a jump is
        never smoothed into a line."""
        c = series["columns"]
        if "x_map" not in c:
            return
        t = _arr(c["t"])
        x, y = _arr(c["x_map"]), _arr(c["y_map"])

        def strips(px, py):
            """Runs of consecutive finite samples, split where a step is too
            long for the car; plus those steps as their own segments."""
            ok = np.isfinite(px) & np.isfinite(py)
            runs, cur, jumps = [], [], []
            for i in range(len(px)):
                if not ok[i]:
                    if len(cur) > 1:
                        runs.append(cur)
                    cur = []
                    continue
                if cur:
                    j = cur[-1]
                    step = math.hypot(px[i] - px[j], py[i] - py[j])
                    # 8 m/s cap on a 20 Hz grid, with room: 0.6 m per sample.
                    if step > max(0.6, 8.0 * (t[i] - t[j])):
                        jumps.append([[px[j], py[j], 0.0], [px[i], py[i], 0.0]])
                        if len(cur) > 1:
                            runs.append(cur)
                        cur = []
                cur.append(i)
            if len(cur) > 1:
                runs.append(cur)
            return [[[px[i], py[i], 0.0] for i in run] for run in runs], jumps

        pose_runs, jumps = strips(x, y)
        if pose_runs:
            self.rec.log(
                "pose2d/pose",
                rr.LineStrips3D(pose_runs, colors=[SERIES[0]], radii=0.03),
                static=True,
            )
        if jumps:
            self.rec.log(
                "pose2d/jumps",
                rr.LineStrips3D(jumps, colors=[STATUS["bad"]], radii=0.05),
                static=True,
            )
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return
        first = np.flatnonzero(ok)[0]
        self.rec.log(
            "pose2d/start",
            rr.Points3D(
                [[x[first], y[first], 0.0]],
                colors=[STATUS["good"]],
                radii=0.12,
                labels=["start"],
            ),
            static=True,
        )
        # The car at the playhead: a marker and its heading, moved over time.
        yaw = _arr(c.get("yaw_map") or [0.0] * len(t))
        ok &= np.isfinite(yaw)
        self.rec.log(
            "pose2d/car",
            rr.Arrows3D(origins=[[0, 0, 0.02]], vectors=[[0.6, 0, 0]],
                        colors=[SERIES[7]], radii=0.04),
            static=True,
        )
        half = yaw[ok] / 2
        rr.send_columns(
            "pose2d/car",
            indexes=[rr.TimeColumn(TIMELINE, duration=t[ok])],
            columns=rr.Transform3D.columns(
                translation=np.column_stack([x[ok], y[ok], np.zeros(ok.sum())]),
                quaternion=np.column_stack(
                    [np.zeros(ok.sum()), np.zeros(ok.sum()), np.sin(half), np.cos(half)]
                ),
            ),
            recording=self.rec,
        )

    def events(self, summary, series):
        c = series["columns"]
        t = _arr(c["t"])
        pts, colors, labels = [], [], []
        for ev in summary.get("events", []):
            if ev["t"] is None:
                continue
            self.at(ev["t"])
            level = {"bad": "ERROR", "warn": "WARN"}.get(ev["severity"], "INFO")
            self.rec.log(
                "events", rr.TextLog(f"[{ev['kind']}] {ev['text']}", level=level)
            )
            if ev["kind"] in ("contact", "graze", "stall", "pose") and "x" in c:
                i = int(np.clip(np.searchsorted(t, ev["t"]), 0, len(t) - 1))
                if c["x"][i] is not None:
                    pts.append([c["x"][i], c["y"][i], 0.25])
                    colors.append(STATUS.get(ev["severity"], GREY))
                    labels.append(f"{ev['kind']} @ {ev['t']:.1f}s")
        if pts:
            self.rec.log(
                "world/events",
                rr.Points3D(pts, colors=colors, radii=0.09, labels=labels),
                static=True,
            )
        # Verdicts, once, at the start: what the Run Lab concluded.
        self.at(summary["window"]["t0"] or 0.0)
        for v in summary.get("verdicts", {}).get("failed", []):
            self.rec.log(
                "events",
                rr.TextLog(f"VERDICT  {v['title']}: {v['detail']}", level="WARN"),
            )

    def logs(self, logs):
        names = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "CRITICAL"}
        for row in logs:
            if row["t"] is None:
                continue
            self.at(row["t"])
            self.rec.log(
                "rosout",
                rr.TextLog(
                    f"[{row['node']}] {row['msg']}",
                    level=names.get(row["level"], "INFO"),
                ),
            )

    # ----------------------------------------------------------- metrics

    def metrics(self, series):
        c = series["columns"]
        t = _arr(c["t"])
        for group, name, column, label, color in CHANNELS:
            if column not in c:
                continue
            v = _arr(c[column])
            ok = np.isfinite(v)
            if not ok.any():
                continue
            path = f"metrics/{group}/{name}"
            self.rec.log(
                path,
                rr.SeriesLines(colors=[color], names=[label], widths=[1.5]),
                static=True,
            )
            rr.send_columns(
                path,
                indexes=[rr.TimeColumn(TIMELINE, duration=t[ok])],
                columns=rr.Scalars.columns(scalars=v[ok]),
                recording=self.rec,
            )
        # The graze band as a flat reference line on the clearance plot.
        if "clearance" in c:
            self.rec.log(
                "metrics/clearance/graze_band",
                rr.SeriesLines(
                    colors=[STATUS["warn"]], names=["graze band 0.12 m"], widths=[1.0]
                ),
                static=True,
            )
            rr.send_columns(
                "metrics/clearance/graze_band",
                indexes=[rr.TimeColumn(TIMELINE, duration=[t[0], t[-1]])],
                columns=rr.Scalars.columns(scalars=[0.12, 0.12]),
                recording=self.rec,
            )

    # ------------------------------------------------------ clouds, camera

    def cloud_frame(self, t, xyz, rgb):
        self.at(t)
        self.rec.log("world/cloud/frame", rr.Points3D(xyz, colors=rgb, radii=0.02))

    def cloud_static(self, entity, xyz, rgb):
        self.rec.log(
            f"world/cloud/{entity}",
            rr.Points3D(xyz, colors=rgb, radii=0.015),
            static=True,
        )

    def image(self, t, jpeg):
        self.at(t)
        self.rec.log("camera", rr.EncodedImage(contents=jpeg, media_type="image/jpeg"))


def blueprint(camera_size=None, *, follow=False):
    """The layout.  ``follow`` opens on the view that rides with the car
    instead of the fixed view of the whole course; both are tabs either way."""
    course = rrb.Spatial3DView(
        origin="world",
        name="Course",
        eye_controls=rrb.EyeControls3D(
            position=[20.0, -15.0, 15.0], look_target=[20.0, 0.5, 0.0], eye_up=[0, 0, 1]
        ),
    )
    # Rooted at the car, so the world is drawn in the car's frame: the eye
    # sits behind and above it and stays there through every move and turn,
    # like a chase camera.  (Tracking an entity from a world-rooted view
    # only follows until the eye is touched or the layout is reloaded.)
    chase = rrb.Spatial3DView(
        origin="world/car",
        name="Follow car",
        # The live cloud only: the accumulated maps bury the car up close.
        contents=["+ /world/**", "- /world/cloud/map", "- /world/cloud/zed_map"],
        eye_controls=rrb.EyeControls3D(
            position=[-3.5, 0.0, 2.0], look_target=[1.5, 0.0, 0.2], eye_up=[0, 0, 1]
        ),
    )
    # Unpinned, the 2D view fills the pane and crops whatever spills over;
    # pinned to the frame it letterboxes the whole image instead.
    bounds = None
    if camera_size:
        bounds = rrb.VisualBounds2D(
            x_range=[0, camera_size[0]], y_range=[0, camera_size[1]]
        )
    plots = rrb.Tabs(
        *[
            rrb.TimeSeriesView(origin=f"metrics/{g}", name=title)
            for g, title in GROUP_TITLES.items()
            if g != "clearance"
        ],
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Tabs(course, chase, active_tab=1 if follow else 0),
                plots,
                row_shares=[3, 2],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera", name="Camera", visual_bounds=bounds),
                rrb.TimeSeriesView(
                    origin="metrics/clearance", name=GROUP_TITLES["clearance"]
                ),
                rrb.Tabs(
                    rrb.TextLogView(origin="events", name="Events"),
                    rrb.TextLogView(origin="rosout", name="ROS log"),
                ),
                row_shares=[3, 2, 2],
            ),
            column_shares=[3, 2],
        ),
        # Paused: the Run Lab's timeline decides where to look first.  Just
        # the play bar -- the Run Lab's own timeline sits under the viewer.
        rrb.TimePanel(timeline=TIMELINE, state="collapsed", play_state="paused"),
        collapse_panels=True,
    )


def pose2d_blueprint(extent=None):
    """The ZED page's layout: the pose alone, looked at straight down with
    x to the right and y up, framed on the pose's extent.  A 3D view rather
    than a 2D one because Rerun's 2D views put y DOWN, which would mirror
    every turn."""
    xmin, xmax, ymin, ymax = extent or (-10.0, 10.0, -10.0, 10.0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    # High enough to see the whole extent through the default ~75 deg FOV.
    height = max(6.0, 0.8 * max(xmax - xmin, ymax - ymin))
    view = rrb.Spatial3DView(
        origin="pose2d",
        name="ZED pose, map frame (red: jumps)",
        eye_controls=rrb.EyeControls3D(
            position=[cx, cy - 0.01, height], look_target=[cx, cy, 0.0], eye_up=[0, 1, 0]
        ),
    )
    return rrb.Blueprint(
        view,
        rrb.TimePanel(timeline=TIMELINE, state="collapsed", play_state="paused"),
        collapse_panels=True,
    )


def _arr(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _ramp(f):
    f = float(np.clip(f, 0.0, 1.0)) * (len(BLUE_RAMP) - 1)
    i = min(int(f), len(BLUE_RAMP) - 2)
    return tuple(
        int(v) for v in BLUE_RAMP[i] + (BLUE_RAMP[i + 1] - BLUE_RAMP[i]) * (f - i)
    )
