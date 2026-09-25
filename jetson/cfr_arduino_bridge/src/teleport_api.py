#!/usr/bin/env python3
"""Small HTTP bridge for simulated robot pose and start signal."""

import json
import math
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Which world to teleport in.  simulation.launch.py sets this from the course
# being launched; the default is the one this started life serving.
WORLD = os.environ.get("CFR_SIM_WORLD", "cfr_speed_course")
MODEL = "slash"
# Fixed by default because the viewer hard-codes it, but overridable: the port
# is plain HTTP and is NOT scoped by ROS_DOMAIN_ID or GZ_PARTITION, so two
# simulations on one machine otherwise share this endpoint and a teleport
# meant for one moves the other one's car.
PORT = int(os.environ.get("CFR_TELEPORT_PORT", "9003"))
# Settling margin above the ground, matching generate_vehicle_model.py.  The
# vehicle model puts wheel bottoms at model-frame z = 0, so anything larger
# drops the car onto its wheels on every teleport.  This was 0.12.
SPAWN_HEIGHT_M = 0.02

MAX_ABS_X = 30.0
MAX_ABS_Y = 20.0
# The Obstacle Course's bridge deck is the highest drivable surface, at
# 25 inches. Anything above this is a caller mistake, not a place the car
# can be put down.
MAX_GROUND_Z = 1.0
# gz service's own wait for Gazebo's reply. 2000ms was fine on a quiet host
# but timed out routinely with a second training container's Gazebo server
# sharing the same CPU (observed: "Service call timed out" mid-run). 20000ms,
# then 40000ms, were each blown past in turn -- the other container's CPU use
# is spiky (observed 237% mid-spike vs. this one's 20%) rather than a steady
# load, so no fixed timeout is truly safe, only progressively less likely to
# lose the race. Tripled from the last value for real margin against a spike.
GZ_SERVICE_TIMEOUT_MS = 120000


class TeleportHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/sim/start-signal":
            self.set_start_signal()
            return
        if self.path != "/api/sim/teleport":
            self.respond(404, {"success": False, "message": "not found"})
            return

        try:
            content_length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(content_length))
            x = float(payload["x"])
            y = float(payload["y"])
            heading = float(payload["heading"])
            # Height of the *ground* under the car, not of the car. Optional
            # and 0.0 by default, so every existing caller is unchanged.
            # The Obstacle Course needs it: its ramp and bridge deck are
            # 0.635 m up, and without this a teleport onto them drops the
            # car through to the floor, which is why RL training could only
            # ever be dealt into the flat part of that course.
            ground_z = float(payload.get("z", 0.0))
        except (KeyError, TypeError, ValueError):
            self.respond(
                400, {"success": False, "message": "x, y, and heading must be numbers"}
            )
            return

        if not all(math.isfinite(value) for value in (x, y, heading, ground_z)):
            self.respond(
                400, {"success": False, "message": "pose values must be finite"}
            )
            return
        if not 0.0 <= ground_z <= MAX_GROUND_Z:
            self.respond(
                400,
                {"success": False, "message": "z is outside the course's height range"},
            )
            return
        if abs(x) > MAX_ABS_X or abs(y) > MAX_ABS_Y:
            self.respond(
                400,
                {
                    "success": False,
                    "message": "position is outside the simulation bounds",
                },
            )
            return

        half_heading = math.radians(heading) / 2.0
        request = (
            f'name: "{MODEL}" '
            f"position {{ x: {x:.9g} y: {y:.9g} z: {ground_z + SPAWN_HEIGHT_M:.9g} }} "
            "orientation { x: 0 y: 0 "
            f"z: {math.sin(half_heading):.17g} w: {math.cos(half_heading):.17g} }}"
        )
        command = [
            "gz",
            "service",
            "-s",
            f"/world/{WORLD}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(GZ_SERVICE_TIMEOUT_MS),
            "--req",
            request,
        ]
        try:
            # A couple seconds above --timeout above: gz service should return
            # on its own within that budget, so this is only a backstop against
            # gz itself hanging, not the normal path out of a slow response.
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=GZ_SERVICE_TIMEOUT_MS / 1000 + 2,
                check=False,
            )
        except FileNotFoundError:
            self.respond(
                502, {"success": False, "message": "Gazebo CLI is unavailable"}
            )
            return
        except subprocess.TimeoutExpired:
            self.respond(
                504, {"success": False, "message": "Gazebo teleport timed out"}
            )
            return
        if result.returncode != 0 or "data: true" not in result.stdout:
            message = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Gazebo rejected teleport"
            )
            self.respond(502, {"success": False, "message": message})
            return

        self.respond(200, {"success": True})

    # The signal arm's two positions, matching red_angle/green_angle in both
    # config/speed_course_layout.yaml and config/obstacle_course_layout.yaml
    # (they agree, and the joint's own limits are 0..pi/2).
    SIGNAL_TOPIC = "/start_signal/arm"
    SIGNAL_RED = 0.0
    SIGNAL_GREEN = 1.57080

    def set_start_signal(self):
        try:
            content_length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(content_length))
            go = payload["go"]
            if type(go) is not bool:
                raise ValueError("go must be a boolean")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.respond(400, {"success": False, "message": str(error)})
            return

        # Which of the optional helpers are actually up.  `ros2 service call`
        # BLOCKS until its service appears, so calling one that is not running
        # burns the whole subprocess timeout and then fails the request -- and
        # it used to fail the request before ever reaching the randomizer, so
        # a stack launched without left_wall_follower could not turn the
        # signal green at all.  Asking first costs one cheap call and makes
        # each helper genuinely optional.
        try:
            listing = subprocess.run(
                ["ros2", "service", "list"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            services = set(listing.stdout.split())
        except FileNotFoundError:
            self.respond(502, {"success": False, "message": "ROS 2 CLI is unavailable"})
            return
        except subprocess.TimeoutExpired:
            services = set()

        def call(service, timeout):
            result = subprocess.run(
                [
                    "ros2",
                    "service",
                    "call",
                    service,
                    "std_srvs/srv/SetBool",
                    "{data: " + ("true" if go else "false") + "}",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode != 0 or "success=True" not in result.stdout:
                raise RuntimeError(
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"{service} failed"
                )

        def sweep_arm():
            """Drive the joint straight over gz transport.

            The randomizer owns the arm normally and ramps it so a detector
            sees the transition rather than a jump.  When it is not running --
            training.launch.py does not start it, and simulation.launch.py
            only does with randomizer:=true -- this is what keeps the viewer's
            button working: the world's JointPositionController is subscribed
            to the topic either way.  Stepped rather than sent once, because a
            one-shot gz publisher can race its own discovery.
            """
            target = self.SIGNAL_GREEN if go else self.SIGNAL_RED
            start = self.SIGNAL_RED if go else self.SIGNAL_GREEN
            steps = 6
            for index in range(1, steps + 1):
                value = start + (target - start) * index / steps
                subprocess.run(
                    [
                        "gz",
                        "topic",
                        "-t",
                        self.SIGNAL_TOPIC,
                        "-m",
                        "gz.msgs.Double",
                        "-p",
                        f"data: {value:.6f}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                time.sleep(0.12)

        used = []
        try:
            # Drive immediately on a manual Go, even if the camera misses the
            # visual transition.  Manual Stop also takes effect immediately.
            for service, timeout in (
                ("/left_wall_follower/manual_start", 10),
                ("/obstacle_randomizer/start_signal", 45),
            ):
                if service in services:
                    call(service, timeout)
                    used.append(service)
            if "/obstacle_randomizer/start_signal" not in services:
                # Nothing owns the arm, so move it here.
                sweep_arm()
                used.append(f"gz {self.SIGNAL_TOPIC}")
        except FileNotFoundError:
            self.respond(502, {"success": False, "message": "ROS 2 CLI is unavailable"})
            return
        except subprocess.TimeoutExpired:
            self.respond(504, {"success": False, "message": "Signal service timed out"})
            return
        except RuntimeError as error:
            self.respond(502, {"success": False, "message": str(error)})
            return
        self.respond(200, {"success": True, "go": go, "via": used})

    def respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), TeleportHandler).serve_forever()
