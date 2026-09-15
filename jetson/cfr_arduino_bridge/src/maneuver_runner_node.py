#!/usr/bin/env python3
"""Run a scripted characterization manoeuvre and record it locally.

Why this exists rather than a human publishing topics by hand: Wi-Fi to the car
drops out at range, so a test that depends on the laptop staying connected is a
test that fails halfway through and wastes the trip.  Once armed, this node
needs nothing off-board.  It drives the profile, records every stream to a run
directory on the Jetson, brings the car back, and writes the result down.

Arming is deliberately awkward.  The node will not move the car until it has
seen E-Stop ASSERTED and then CLEARED while auto is armed.  That sequence can
only happen if the operator is holding a working, connected E-Stop, which is
exactly the precondition worth enforcing before anything drives itself.

This node has NO safety authority.  The E-Stop is the safety device; this is an
ordinary autonomy client, subject to it, to the Arduino's 200 ms watchdog, and
to the AUTO_ARMED -> AUTO_ACTIVE handshake.  Re-asserting E-Stop aborts a run.

Usage is through characterize.launch.py, which picks the run directory and
points the bridge's serial traces into it:

    ros2 launch cfr_arduino_bridge characterize.launch.py profile:=coastdown
"""

import csv
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import rclpy
import yaml
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cfr_interfaces.msg import ArduinoStatus, DriveCommand

# Pack voltage endpoints the firmware's BATTERY_LEVEL maps between, so a 0-255
# level can be reported as volts.  Exact millivolts are in the D, debug line;
# this is for at-a-glance metadata and matches what the E-Stop TUI shows.
BATTERY_EMPTY_MV = 10000.0
BATTERY_FULL_MV = 12600.0

# Topics recorded into the bag.  The CSV is the primary analysis artifact; the
# bag is the raw backup for anything the CSV does not carry.
BAG_TOPICS = [
    "/drive_cmd",
    "/cmd_vel",
    "/arduino_bridge/status",
    "/zed/zed_node/odom",
    "/zed/zed_node/imu/data",
]

CSV_COLUMNS = [
    "t_ros",
    "t_elapsed",
    "phase",
    "step_index",
    "step_label",
    "step_phase",
    "cmd_steering",
    "cmd_velocity",
    "auto_ready",
    "mode",
    "link_ok",
    "estop",
    "gains_applied",
    "battery_level",
    "battery_volts",
    "rpm",
    "wheel_rpm",
    "speed",
    "target_speed",
    "throttle_us",
    "odom_valid",
    "odom_x",
    "odom_y",
    "odom_yaw",
    "odom_vx",
    "odom_wz",
    "dist_along",
    "dist_total",
]


def yaw_from_quaternion(w, x, y, z):
    """Planar yaw, matching YawFromQuaternion in path_geometry.cpp."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def battery_volts(level):
    """Approximate pack volts from the 0-255 BATTERY_LEVEL code."""
    span = BATTERY_FULL_MV - BATTERY_EMPTY_MV
    return (BATTERY_EMPTY_MV + (level / 255.0) * span) / 1000.0


class Profile:
    """A parsed manoeuvre profile.

    Raises ValueError with a message aimed at whoever is standing next to the
    car, because that is where a broken profile gets discovered.
    """

    def __init__(self, data, path):
        if not isinstance(data, dict):
            raise ValueError(f"{path}: profile must be a YAML mapping")
        self.path = path
        self.name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
        self.description = (data.get("description") or "").strip()
        self.gains = dict(data.get("gains") or {})
        # `none` records without ever setting auto_ready, so the car cannot move.
        # A five minute stationary odometry recording has no business arming the
        # ESC, and making that the profile's decision keeps it out of the
        # operator's hands.
        self.arming = str(data.get("arming", "estop_cycle"))
        if self.arming not in ("estop_cycle", "none"):
            raise ValueError(f"{path}: arming must be estop_cycle or none")
        if self.arming == "none":
            moving = [
                step
                for step in (data.get("steps") or [])
                if isinstance(step, dict) and float(step.get("velocity", 0.0)) != 0.0
            ]
            if moving:
                raise ValueError(
                    f"{path}: arming is none but a step commands a nonzero velocity"
                )

        limits = dict(data.get("limits") or {})
        self.max_distance = float(limits.get("max_distance", 40.0))
        self.max_duration = float(limits.get("max_duration", 180.0))
        self.max_speed = float(limits.get("max_speed", 3.5))
        if (
            self.max_distance <= 0.0
            or self.max_duration <= 0.0
            or self.max_speed <= 0.0
        ):
            raise ValueError(f"{path}: limits must all be positive")

        steps = data.get("steps") or []
        if not steps:
            raise ValueError(f"{path}: profile has no steps")
        self.steps = []
        for index, raw in enumerate(steps):
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: step {index} must be a mapping")
            hold = float(raw.get("hold", 0.0))
            if hold <= 0.0:
                raise ValueError(f"{path}: step {index} needs a positive hold")
            velocity = float(raw.get("velocity", 0.0))
            if abs(velocity) > self.max_speed + 1e-9:
                raise ValueError(
                    f"{path}: step {index} velocity {velocity} exceeds the "
                    f"profile max_speed {self.max_speed}"
                )
            steering = float(raw.get("steering", 0.0))
            if not -1.0 <= steering <= 1.0:
                raise ValueError(
                    f"{path}: step {index} steering must be within [-1, 1]"
                )
            self.steps.append(
                {
                    "hold": hold,
                    "steering": steering,
                    "velocity": velocity,
                    "label": str(raw.get("label", f"step{index}")),
                    "gains": dict(raw.get("gains") or {}),
                }
            )

        ret = dict(data.get("return") or {})
        self.return_mode = str(ret.get("mode", "none"))
        if self.return_mode not in ("none", "reverse_to_start"):
            raise ValueError(f"{path}: return mode must be none or reverse_to_start")
        self.return_speed = abs(float(ret.get("speed", 1.2)))
        self.return_tolerance = float(ret.get("tolerance", 1.0))
        self.return_timeout = float(ret.get("timeout", 90.0))

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle), path)


class ManeuverRunner(Node):
    # Phases.  The arming chain is the safety interlock; everything before
    # RUNNING holds the car at neutral.
    WAIT_LINK = "wait_link"
    WAIT_ESTOP_ASSERTED = "wait_estop_asserted"
    WAIT_ESTOP_CLEARED = "wait_estop_cleared"
    ARMING = "arming"
    RUNNING = "running"
    RETURNING = "returning"
    STOPPING = "stopping"
    FINISHED = "finished"

    def __init__(self):
        super().__init__("maneuver_runner")

        profile_path = self.declare_parameter("profile", "").value
        self.run_dir = self.declare_parameter("run_dir", "").value
        self.control_rate_hz = float(
            self.declare_parameter("control_rate_hz", 50.0).value
        )
        self.countdown = float(self.declare_parameter("countdown", 3.0).value)
        self.odom_timeout = float(self.declare_parameter("odom_timeout", 0.5).value)
        self.gains_timeout = float(self.declare_parameter("gains_timeout", 8.0).value)
        self.require_estop_cycle = bool(
            self.declare_parameter("require_estop_cycle", True).value
        )
        self.record_bag = bool(self.declare_parameter("record_bag", True).value)
        self.bridge_node = self.declare_parameter(
            "bridge_node", "/arduino_bridge"
        ).value
        self.operator = self.declare_parameter("operator", "").value
        self.surface = self.declare_parameter("surface", "asphalt").value
        self.notes = self.declare_parameter("notes", "").value
        overrides = self.declare_parameter("gain_overrides", "").value

        if not profile_path:
            raise RuntimeError("the profile parameter is required")
        self.profile = Profile.load(profile_path)
        self._apply_gain_overrides(overrides)

        if not self.run_dir:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.run_dir = os.path.expanduser(f"~/cfr_runs/{stamp}_{self.profile.name}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.status = None
        self.status_time = None
        self.odom = None
        self.odom_time = None

        self.phase = self.WAIT_LINK
        self.phase_entered = time.monotonic()
        self.started_monotonic = None
        self.step_index = 0
        self.step_started = None
        self.step_phase = "hold"
        self.gains_requested_at = None
        self.gains_pending = False
        self.gains_future = None
        self.gains_error = None
        self.requested_gain_names = []
        self.command = (0.0, 0.0)  # steering, velocity actually published
        self.auto_ready = False
        self.origin = None  # (x, y, yaw) captured when the run starts
        self.result = None
        self.result_detail = ""
        self.battery_start = None
        self.battery_end = None
        self.peak_distance = 0.0

        self.publisher = self.create_publisher(
            DriveCommand, "drive_cmd", qos_profile_sensor_data
        )
        self.create_subscription(ArduinoStatus, "status", self._on_status, 10)
        self.create_subscription(
            Odometry, "odom", self._on_odom, qos_profile_sensor_data
        )
        self.parameter_client = self.create_client(
            SetParameters, f"{self.bridge_node}/set_parameters"
        )

        self.log_path = os.path.join(self.run_dir, "runner.log")
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        self.csv_file = open(
            os.path.join(self.run_dir, "telemetry.csv"),
            "w",
            encoding="utf-8",
            newline="",
        )
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(CSV_COLUMNS)

        self.bag_process = self._start_bag()
        self._write_metadata()

        self.say(f'profile "{self.profile.name}" loaded from {self.profile.path}')
        if self.profile.description:
            self.say(self.profile.description)
        self.say(f"recording to {self.run_dir}")
        self.say(
            f"limits: {self.profile.max_distance:.0f} m, "
            f"{self.profile.max_duration:.0f} s, {self.profile.max_speed:.1f} m/s"
        )
        self.say("waiting for the Arduino link")

        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self._tick)

    def _apply_gain_overrides(self, overrides):
        """Fold `name=value` overrides into the profile's gain block.

        A gain sweep is several runs of one profile with one number changed, and
        editing YAML on a laptop with no network, standing next to the car, is
        how the wrong gains end up in a run directory labelled as the right ones.
        Overrides are recorded in metadata.yaml so the run is still self-describing.
        """
        self.gain_overrides = {}
        for token in (overrides or "").replace(",", " ").split():
            if "=" not in token:
                raise RuntimeError(f'gain override "{token}" is not name=value')
            name, _, raw = token.partition("=")
            name = name.strip()
            raw = raw.strip()
            if raw.lower() in ("true", "false"):
                value = raw.lower() == "true"
            else:
                try:
                    value = float(raw)
                except ValueError as error:
                    raise RuntimeError(
                        f'gain override "{token}" is not a number'
                    ) from error
            self.gain_overrides[name] = value
        if self.gain_overrides:
            self.profile.gains.update(self.gain_overrides)

    # ---------------------------------------------------------------- logging

    def say(self, message):
        """Log to the console and into the run directory.

        The run log matters more than the console here: by the time anyone reads
        it the laptop may have been out of range for the whole test.
        """
        self.get_logger().info(message)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log_file.write(f"{stamp} {message}\n")
        self.log_file.flush()

    def warn(self, message):
        self.get_logger().warn(message)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log_file.write(f"{stamp} WARN {message}\n")
        self.log_file.flush()

    # ------------------------------------------------------------ subscribers

    def _on_status(self, msg):
        self.status = msg
        self.status_time = time.monotonic()
        if msg.link_ok:
            if self.battery_start is None:
                self.battery_start = msg.battery_level
            self.battery_end = msg.battery_level

    def _on_odom(self, msg):
        self.odom = msg
        self.odom_time = time.monotonic()

    # ------------------------------------------------------------- recording

    def _start_bag(self):
        if not self.record_bag:
            return None
        if shutil.which("ros2") is None:
            self.warn(
                "ros2 not on PATH, skipping bag recording (telemetry.csv is unaffected)"
            )
            return None
        command = [
            "ros2",
            "bag",
            "record",
            "-o",
            os.path.join(self.run_dir, "bag"),
        ] + BAG_TOPICS
        try:
            # Own process group so the bag can be stopped without signalling the
            # whole launch, and so a crashed runner does not orphan it.
            return subprocess.Popen(
                command,
                stdout=open(
                    os.path.join(self.run_dir, "bag.log"), "w", encoding="utf-8"
                ),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            self.warn(
                f"could not start bag recording ({error}); telemetry.csv is unaffected"
            )
            return None

    def _stop_bag(self):
        if self.bag_process is None or self.bag_process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGINT)
            self.bag_process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.bag_process.pid), signal.SIGKILL)
            except OSError:
                pass

    def _write_metadata(self, final=False):
        metadata = {
            "profile": self.profile.name,
            "profile_path": self.profile.path,
            "description": self.profile.description,
            "run_dir": self.run_dir,
            "started_utc": datetime.now(timezone.utc).isoformat()
            if not final
            else self.started_utc,
            "operator": self.operator,
            "surface": self.surface,
            "notes": self.notes,
            "git_sha": self._git_sha(),
            "gain_overrides": self.gain_overrides,
            "profile_gains": self.profile.gains,
            "arming": self.profile.arming,
            "limits": {
                "max_distance_m": self.profile.max_distance,
                "max_duration_s": self.profile.max_duration,
                "max_speed_mps": self.profile.max_speed,
            },
        }
        if not final:
            self.started_utc = metadata["started_utc"]
            metadata["bridge_parameters"] = self._dump_bridge_parameters()
        else:
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["result"] = self.result
            metadata["result_detail"] = self.result_detail
            metadata["peak_distance_m"] = round(self.peak_distance, 3)
            metadata["bridge_parameters"] = self.bridge_parameters
            if self.battery_start is not None:
                metadata["battery"] = {
                    "level_start": int(self.battery_start),
                    "level_end": int(self.battery_end),
                    "volts_start_approx": round(battery_volts(self.battery_start), 2),
                    "volts_end_approx": round(battery_volts(self.battery_end), 2),
                    "note": "Approximate. Exact millivolts are in the D, lines of arduino_rx.log.",
                }
        path = os.path.join(self.run_dir, "metadata.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False, default_flow_style=False)

    def _git_sha(self):
        """Which build produced this run.  Best effort; a run without it is fine.

        Tries several roots because this file normally runs from the INSTALL
        space, which is not a git checkout - so asking git about __file__'s
        directory alone would come back empty on exactly the runs that matter.
        """
        candidates = [
            os.path.dirname(os.path.abspath(__file__)),
            os.getcwd(),
            os.path.expanduser("~/software"),
        ]
        for root in candidates:
            try:
                output = subprocess.run(
                    ["git", "-C", root, "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if output.returncode == 0 and output.stdout.strip():
                return output.stdout.strip()
        return ""

    def _dump_bridge_parameters(self):
        """Snapshot the bridge's parameters so a run is reproducible.

        Best effort: a run with no parameter dump is still a usable run, and
        failing the test over it would be the wrong trade in the field.
        """
        self.bridge_parameters = {}
        if shutil.which("ros2") is None:
            return self.bridge_parameters
        try:
            output = subprocess.run(
                ["ros2", "param", "dump", self.bridge_node],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if output.returncode == 0 and output.stdout.strip():
                self.bridge_parameters = yaml.safe_load(output.stdout) or {}
        except (OSError, subprocess.TimeoutExpired, yaml.YAMLError) as error:
            self.warn(f"could not dump bridge parameters: {error}")
        return self.bridge_parameters

    # --------------------------------------------------------------- helpers

    def _pose(self):
        if self.odom is None:
            return None
        position = self.odom.pose.pose.position
        orientation = self.odom.pose.pose.orientation
        return (
            position.x,
            position.y,
            yaw_from_quaternion(
                orientation.w, orientation.x, orientation.y, orientation.z
            ),
        )

    def _distances(self):
        """Displacement from the arm point: along the initial heading, and total.

        `along` is what the return leg drives to zero.  Projecting rather than
        using the magnitude means lateral drift does not make the car think it
        still has ground to cover once it is back level with the start.
        """
        pose = self._pose()
        if pose is None or self.origin is None:
            return (0.0, 0.0)
        dx = pose[0] - self.origin[0]
        dy = pose[1] - self.origin[1]
        along = dx * math.cos(self.origin[2]) + dy * math.sin(self.origin[2])
        return (along, math.hypot(dx, dy))

    def _odom_fresh(self):
        return (
            self.odom_time is not None
            and (time.monotonic() - self.odom_time) < self.odom_timeout
        )

    def _link_ok(self):
        return (
            self.status is not None
            and self.status.link_ok
            and self.status_time is not None
            and (time.monotonic() - self.status_time) < 1.0
        )

    def _set_phase(self, phase):
        self.phase = phase
        self.phase_entered = time.monotonic()

    def _request_gains(self, gains):
        """Push a step's gains onto the bridge, which relays them to the Arduino.

        Returns False if the service is not up; the caller treats that as fatal
        because a profile whose gains never took is not the profile that was run.
        """
        # Always stamp the request time, even for a step with no gains: callers
        # use it to tell "not asked yet" from "asked and waiting", and a profile
        # with an empty gains block would otherwise never leave arming.
        self.gains_requested_at = time.monotonic()
        self.gains_future = None
        self.gains_error = None
        self.gains_pending = False
        if not gains:
            return True
        if not self.parameter_client.service_is_ready():
            if not self.parameter_client.wait_for_service(timeout_sec=5.0):
                return False
        request = SetParameters.Request()
        for name, value in sorted(gains.items()):
            parameter = Parameter()
            parameter.name = name
            if isinstance(value, bool):
                parameter.value = ParameterValue(
                    type=ParameterType.PARAMETER_BOOL, bool_value=value
                )
            else:
                parameter.value = ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)
                )
            request.parameters.append(parameter)
        self.requested_gain_names = [parameter.name for parameter in request.parameters]
        self.gains_future = self.parameter_client.call_async(request)
        self.gains_pending = True
        return True

    def _gains_settled(self):
        """True once the Arduino is running the gains we last asked for.

        Two things have to go right, and checking only the second is a trap the
        first version of this fell into.  The bridge REJECTS a gain set the
        firmware would not accept, and a rejected set leaves its sequence number
        untouched - so gains_applied stays true and a run would quietly continue
        on the previous gains, mislabelled as the new ones.  So check the service
        result first, then the acknowledgement.

        The 0.3 s floor is there because gains_applied is still true for the
        cycle or two between the service call and the bridge bumping its
        sequence number; without it a step would start timing against the gains
        it is replacing.
        """
        if not self.gains_pending:
            return True
        if self.gains_future is None or not self.gains_future.done():
            return False
        if self.gains_error is None:
            response = self.gains_future.result()
            if response is None:
                self.gains_error = "parameter service call failed"
            else:
                rejected = [
                    f"{name} ({result.reason or 'rejected'})"
                    for name, result in zip(self.requested_gain_names, response.results)
                    if not result.successful
                ]
                if rejected:
                    self.gains_error = "bridge rejected " + ", ".join(rejected)
        if self.gains_error is not None:
            return False
        if (time.monotonic() - self.gains_requested_at) < 0.3:
            return False
        if self.status is not None and self.status.gains_applied:
            self.gains_pending = False
            return True
        return False

    def _abort(self, reason):
        self.warn(f"ABORT: {reason}")
        self.result = "aborted"
        self.result_detail = reason
        self.command = (0.0, 0.0)
        self._set_phase(self.STOPPING)

    def _finish(self, reason):
        self.say(f"complete: {reason}")
        if self.result is None:
            self.result = "ok"
            self.result_detail = reason
        self.command = (0.0, 0.0)
        self._set_phase(self.STOPPING)

    # ------------------------------------------------------------ main update

    def _tick(self):
        now = time.monotonic()

        if self.phase in (self.RUNNING, self.RETURNING):
            self._check_aborts(now)

        handler = {
            self.WAIT_LINK: self._tick_wait_link,
            self.WAIT_ESTOP_ASSERTED: self._tick_wait_estop_asserted,
            self.WAIT_ESTOP_CLEARED: self._tick_wait_estop_cleared,
            self.ARMING: self._tick_arming,
            self.RUNNING: self._tick_running,
            self.RETURNING: self._tick_returning,
            self.STOPPING: self._tick_stopping,
            self.FINISHED: lambda now: None,
        }[self.phase]
        handler(now)

        self._publish()
        self._record()

    def _check_aborts(self, now):
        if not self._link_ok():
            self._abort("Arduino link lost")
            return
        # A non-arming profile is a stationary recording: it never enters
        # AUTO_ACTIVE, and E-Stop being asserted throughout is the normal, safe
        # state for it rather than something to abort on.
        if self.profile.arming != "none":
            if self.status.estop or self.status.mode == ArduinoStatus.MODE_ESTOP:
                self._abort("E-Stop asserted")
                return
            if self.status.mode != ArduinoStatus.MODE_AUTO_ACTIVE:
                self._abort(f"left AUTO_ACTIVE (mode {self.status.mode})")
                return
        if not self._odom_fresh():
            self._abort(f"odometry stale (> {self.odom_timeout:.2f} s)")
            return
        if self.started_monotonic is not None:
            elapsed = now - self.started_monotonic
            if elapsed > self.profile.max_duration:
                self._abort(f"exceeded max_duration {self.profile.max_duration:.0f} s")
                return
        _, total = self._distances()
        self.peak_distance = max(self.peak_distance, total)
        if total > self.profile.max_distance:
            self._abort(
                f"exceeded max_distance {self.profile.max_distance:.0f} m "
                f"(reached {total:.1f} m)"
            )

    def _tick_wait_link(self, now):
        self.auto_ready = False
        self.command = (0.0, 0.0)
        if self._link_ok():
            self.say("Arduino link up")
            if self.profile.arming == "none":
                self.say("profile does not arm: recording only, the car will not move")
                self._set_phase(self.ARMING)
            elif self.require_estop_cycle:
                self.say(
                    "ARM: assert E-Stop now (press the button), then clear it to start"
                )
                self._set_phase(self.WAIT_ESTOP_ASSERTED)
            else:
                self.warn(
                    "require_estop_cycle is false - arming without the E-Stop interlock"
                )
                self._set_phase(self.ARMING)

    def _tick_wait_estop_asserted(self, now):
        self.auto_ready = False
        self.command = (0.0, 0.0)
        if self.status.estop or self.status.mode == ArduinoStatus.MODE_ESTOP:
            self.say("E-Stop asserted - now clear it to start the run")
            self._set_phase(self.WAIT_ESTOP_CLEARED)
        elif (now - self.phase_entered) > 10.0:
            self.phase_entered = now
            self.say("still waiting for E-Stop to be ASSERTED")

    def _tick_wait_estop_cleared(self, now):
        # auto_ready goes true here so the Arduino can complete its
        # AUTO_ARMED -> AUTO_ACTIVE handshake the moment E-Stop clears.  Steering
        # and speed stay at neutral, which that handshake requires anyway.
        self.auto_ready = True
        self.command = (0.0, 0.0)
        if not self.status.estop and self.status.mode != ArduinoStatus.MODE_ESTOP:
            if not self.status.auto_arm:
                if (now - self.phase_entered) > 5.0:
                    self.phase_entered = now
                    self.warn(
                        "E-Stop is clear but auto is not armed - set Auto Arm on the controller"
                    )
                return
            self.say("E-Stop cleared with auto armed - arming")
            self._set_phase(self.ARMING)

    def _tick_arming(self, now):
        self.auto_ready = self.profile.arming != "none"
        self.command = (0.0, 0.0)
        if (
            self.profile.arming != "none"
            and self.status.mode != ArduinoStatus.MODE_AUTO_ACTIVE
        ):
            if (now - self.phase_entered) > 5.0:
                self.phase_entered = now
                self.warn(
                    f"waiting for AUTO_ACTIVE, Arduino is in mode {self.status.mode}"
                )
            return
        if not self._odom_fresh():
            if (now - self.phase_entered) > 5.0:
                self.phase_entered = now
                self.warn("waiting for odometry before starting")
            return

        # The profile's gains have to be in place before the first step, and the
        # countdown is free time in which to do it.
        if self.gains_requested_at is None:
            if not self._request_gains(self.profile.gains):
                self._abort("bridge parameter service unavailable")
                return
            self.say(f"applying profile gains, starting in {self.countdown:.0f} s")
            return
        if not self._gains_settled():
            if self.gains_error is not None:
                self._abort(f"profile gains refused: {self.gains_error}")
            elif (now - self.gains_requested_at) > self.gains_timeout:
                self._abort("Arduino never acknowledged the profile gains")
            return
        if (now - self.gains_requested_at) < self.countdown:
            return

        self.origin = self._pose()
        self.started_monotonic = now
        self.step_index = 0
        self.step_started = now
        self.step_phase = "hold"
        self.say(f"RUN START: {len(self.profile.steps)} steps")
        self._set_phase(self.RUNNING)

    def _tick_running(self, now):
        if self.phase != self.RUNNING:
            return  # an abort fired during _check_aborts
        step = self.profile.steps[self.step_index]

        if self.step_phase == "gains_wait":
            # Hold the previous step's command while the new gains land.  The
            # car keeps moving, so a staircase profile does not stop and restart
            # between points; the CSV marks the window so analysis can drop it.
            if self._gains_settled():
                self.step_phase = "hold"
                self.step_started = now
                self.say(
                    f"step {self.step_index}/{len(self.profile.steps) - 1}: {step['label']}"
                )
            elif self.gains_error is not None:
                self._abort(
                    f"gains refused for step {self.step_index}: {self.gains_error}"
                )
            elif (now - self.gains_requested_at) > self.gains_timeout:
                self._abort(
                    f"Arduino never acknowledged gains for step {self.step_index}"
                )
            return

        self.command = (step["steering"], step["velocity"])
        if (now - self.step_started) < step["hold"]:
            return

        self.step_index += 1
        if self.step_index >= len(self.profile.steps):
            if self.profile.return_mode == "reverse_to_start":
                along, _ = self._distances()
                self.say(f"steps complete at {along:.1f} m out, reversing back")
                self._set_phase(self.RETURNING)
            else:
                self._finish("all steps complete")
            return

        next_step = self.profile.steps[self.step_index]
        if next_step["gains"]:
            if not self._request_gains(next_step["gains"]):
                self._abort("bridge parameter service unavailable")
                return
            self.step_phase = "gains_wait"
        else:
            self.step_started = now
            self.say(
                f"step {self.step_index}/{len(self.profile.steps) - 1}: {next_step['label']}"
            )

    def _tick_returning(self, now):
        if self.phase != self.RETURNING:
            return
        along, _ = self._distances()
        if along <= self.profile.return_tolerance:
            self._finish(f"returned to within {along:.2f} m of the start")
            return
        if (now - self.phase_entered) > self.profile.return_timeout:
            self.warn(
                f"return leg timed out {along:.1f} m from the start - stopping there"
            )
            self._finish("return leg timed out")
            return
        # Straight-line reverse, steering held at centre.  No steering under
        # odometry and no turn-around, so the car never drives forward at
        # whoever is holding the E-Stop.
        self.command = (0.0, -self.profile.return_speed)

    def _tick_stopping(self, now):
        self.command = (0.0, 0.0)
        # Hold a zero speed target briefly before dropping auto_ready, so the
        # stop is a commanded one rather than a watchdog timeout.
        if (now - self.phase_entered) > 1.0:
            self.auto_ready = False
        if (now - self.phase_entered) > 2.0:
            self._set_phase(self.FINISHED)

    # ------------------------------------------------------------ outputs

    def _publish(self):
        message = DriveCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.auto_ready = self.auto_ready and self.profile.arming != "none"
        message.steering = float(self.command[0])
        message.velocity = float(self.command[1])
        self.publisher.publish(message)

    def _record(self):
        status = self.status
        odom = self.odom
        along, total = self._distances()
        step_label = ""
        if self.phase == self.RUNNING and self.step_index < len(self.profile.steps):
            step_label = self.profile.steps[self.step_index]["label"]

        row = [
            f"{self.get_clock().now().nanoseconds / 1e9:.6f}",
            f"{(time.monotonic() - self.started_monotonic):.4f}"
            if self.started_monotonic
            else "",
            self.phase,
            self.step_index if self.phase == self.RUNNING else "",
            step_label,
            self.step_phase if self.phase == self.RUNNING else "",
            f"{self.command[0]:.4f}",
            f"{self.command[1]:.4f}",
            int(self.auto_ready),
        ]
        if status is not None:
            row += [
                status.mode,
                int(status.link_ok),
                int(status.estop),
                int(status.gains_applied),
                status.battery_level,
                f"{battery_volts(status.battery_level):.2f}",
                status.rpm,
                f"{status.wheel_rpm:.2f}",
                f"{status.speed:.4f}",
                f"{status.target_speed:.4f}",
                status.throttle_us,
            ]
        else:
            row += [""] * 11
        if odom is not None:
            pose = self._pose()
            row += [
                int(self._odom_fresh()),
                f"{pose[0]:.4f}",
                f"{pose[1]:.4f}",
                f"{pose[2]:.5f}",
                f"{odom.twist.twist.linear.x:.4f}",
                f"{odom.twist.twist.angular.z:.5f}",
            ]
        else:
            row += [0, "", "", "", "", ""]
        row += [f"{along:.4f}", f"{total:.4f}"]
        self.csv_writer.writerow(row)

    def done(self):
        return self.phase == self.FINISHED

    def shutdown(self):
        if self.result is None:
            self.result = "incomplete"
            self.result_detail = "runner exited before the profile finished"
        self._stop_bag()
        try:
            self._write_metadata(final=True)
        finally:
            self.csv_file.close()
            self.say(f"run directory: {self.run_dir}  result: {self.result}")
            self.log_file.close()


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = ManeuverRunner()
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if node is not None:
            node.warn("interrupted from the console")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
