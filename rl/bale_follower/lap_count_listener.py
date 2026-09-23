#!/usr/bin/env python3
"""Print each lap crossing the REAL lap_counter node reports, with its timing.

This is the ground truth for "how fast did it actually lap" -- the same
LapCount message a competition run and the field-card procedure both read --
as opposed to path_racer.py's own arc-length bookkeeping, which this is meant
to cross-check.
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node

from cfr_interfaces.msg import LapCount


class Listener(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("lap_count_listener")
        self.last_lap = 0
        self.last_crossing_wall = time.monotonic()
        self.create_subscription(LapCount, topic, self.on_count, 10)

    def on_count(self, msg: LapCount) -> None:
        if msg.laps > self.last_lap:
            now = time.monotonic()
            self.get_logger().info(
                f"OFFICIAL LAP {msg.laps}/{msg.target}: "
                f"{now - self.last_crossing_wall:.2f} s wall, "
                f"lap_distance {msg.lap_distance:.1f} m, "
                f"rejected {msg.rejected}, loop_closures {msg.loop_closures}"
            )
            self.last_lap = msg.laps
            self.last_crossing_wall = now
        if msg.done:
            self.get_logger().info(f"DONE: {msg.laps} laps completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/lap_counter/count")
    args = parser.parse_args()
    rclpy.init()
    node = Listener(args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
