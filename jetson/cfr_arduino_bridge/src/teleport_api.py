#!/usr/bin/env python3
"""Small HTTP bridge for setting the simulated robot pose."""

import json
import math
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Which world to teleport in.  simulation.launch.py sets this from the course
# being launched; the default is the one this started life serving.
WORLD = os.environ.get("CFR_SIM_WORLD", "cfr_speed_course")
MODEL = "slash"
PORT = 9003
MAX_ABS_X = 30.0
MAX_ABS_Y = 20.0


class TeleportHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/sim/teleport":
            self.respond(404, {"success": False, "message": "not found"})
            return

        try:
            content_length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(content_length))
            x = float(payload["x"])
            y = float(payload["y"])
            heading = float(payload["heading"])
        except (KeyError, TypeError, ValueError):
            self.respond(
                400, {"success": False, "message": "x, y, and heading must be numbers"}
            )
            return

        if not all(math.isfinite(value) for value in (x, y, heading)):
            self.respond(
                400, {"success": False, "message": "pose values must be finite"}
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
            f"position {{ x: {x:.9g} y: {y:.9g} z: 0.12 }} "
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
            "2000",
            "--req",
            request,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=3, check=False
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
