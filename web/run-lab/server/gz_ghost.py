#!/usr/bin/env python3
"""Drive the Gazebo car along a recorded run -- a kinematic "ghost" replay.

    python3 gz_ghost.py --poses poses.json [--rate 1.0] [--start 0] [--loop]

Runs under the SYSTEM ROS Python (rclpy, ros_gz_interfaces), started by the
Run Lab's replay manager next to a speed_course.launch.py it owns.  The poses
are the run's TRACK-frame poses, which is Gazebo's world frame, so the car in
Gazebo sits exactly where the real car sat on the course.

Poses go in through `ros_gz_bridge`'s bridged /world/<w>/set_pose service
rather than the `gz service` CLI: the CLI starts a Ruby interpreter per call,
~0.3 s, which caps a replay at a jerky 3 Hz.  At most two requests are in
flight; a slow Gazebo drops frames instead of falling ever further behind.

Nothing here commands the drive: no /drive_cmd publisher exists in the replay
launch, so sim_vehicle_node holds neutral and the teleports are the only thing
moving the car.  Physics keeps running, which is what lets the car settle onto
its wheels between frames.

Control: a tiny HTTP endpoint on --control-port accepts
    POST /seek {"t": seconds}   POST /pause   POST /play   POST /rate {"rate": 2}
    GET  /state
so the browser's timeline can steer the Gazebo replay.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose

SPAWN_Z = 0.02  # teleport_api.SPAWN_HEIGHT_M: wheels at model z = 0


class Ghost(Node):
    def __init__(self, args):
        super().__init__("runlab_ghost")
        data = json.loads(open(args.poses).read())
        self.t = data["t"]
        self.x = data["x"]
        self.y = data["y"]
        self.yaw = data["yaw"]
        self.model = args.model
        self.client = self.create_client(SetEntityPose, f"/world/{args.world}/set_pose")
        self.rate = args.rate
        self.loop = args.loop
        self.playing = True
        self.pos = args.start  # seconds of run time
        self.wall = time.monotonic()
        self.in_flight = 0
        self.sent = 0
        self.dropped = 0
        self.lock = threading.Lock()
        self.create_timer(1.0 / args.hz, self.tick)

    def state(self):
        return {
            "t": round(self.pos, 3),
            "t_end": self.t[-1] if self.t else 0,
            "playing": self.playing,
            "rate": self.rate,
            "sent": self.sent,
            "dropped": self.dropped,
            "service_ready": self.client.service_is_ready(),
        }

    def sample(self, when):
        """Nearest valid pose at run time `when` (the series has gaps)."""
        import bisect

        i = bisect.bisect_left(self.t, when)
        # Nearest, not the next: the clock is a float and bisect_left alone
        # rounds up, which put the ghost one sample (50 ms) ahead.
        if i > 0 and (i == len(self.t) or when - self.t[i - 1] <= self.t[i] - when):
            i -= 1
        for j in (i, i - 1, i + 1, i - 2, i + 2):
            if 0 <= j < len(self.t) and self.x[j] is not None and self.y[j] is not None:
                return self.x[j], self.y[j], self.yaw[j] or 0.0
        return None

    def tick(self):
        now = time.monotonic()
        with self.lock:
            if self.playing:
                self.pos += (now - self.wall) * self.rate
            self.wall = now
            if self.t and self.pos > self.t[-1]:
                if self.loop:
                    self.pos = self.t[0]
                else:
                    self.pos = self.t[-1]
                    self.playing = False
            when = self.pos
        if not self.client.service_is_ready():
            return
        if self.in_flight >= 2:
            self.dropped += 1
            return
        pose = self.sample(when)
        if pose is None:
            return
        x, y, yaw = pose
        req = SetEntityPose.Request()
        req.entity.name = self.model
        req.entity.type = 2  # MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = SPAWN_Z
        req.pose.orientation.z = math.sin(yaw / 2)
        req.pose.orientation.w = math.cos(yaw / 2)
        self.in_flight += 1
        self.sent += 1
        future = self.client.call_async(req)
        future.add_done_callback(self._done)

    def _done(self, _future):
        self.in_flight = max(0, self.in_flight - 1)


def serve_control(ghost, port):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(ghost.state())

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with ghost.lock:
                if self.path == "/seek":
                    ghost.pos = float(body.get("t", 0.0))
                elif self.path == "/pause":
                    ghost.playing = False
                elif self.path == "/play":
                    ghost.playing = True
                elif self.path == "/rate":
                    ghost.rate = max(0.05, min(8.0, float(body.get("rate", 1.0))))
                else:
                    return self._reply({"error": "unknown"}, 404)
            self._reply(ghost.state())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--world", default="cfr_speed_course")
    ap.add_argument("--model", default="slash")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--control-port", type=int, default=9031)
    args = ap.parse_args()
    rclpy.init()
    ghost = Ghost(args)
    serve_control(ghost, args.control_port)
    try:
        rclpy.spin(ghost)
    except KeyboardInterrupt:
        pass
    finally:
        ghost.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
