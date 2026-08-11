#!/usr/bin/env python3
"""Interactive TUI to build and run a DrivePath goal for path_follower_node.

Build a list of STRAIGHT/TURN segments (e.g. the L-shape: 5 ft, 90 deg right,
2 ft), then send it as one DrivePath goal and watch it execute with live
feedback.  Needs the workspace sourced first:

    source ~/ros2_ws/install/setup.bash
    python3 path_tui.py

Ctrl-C (or 'q') while a path is executing cancels the goal before exiting --
this tool is the only thing telling the car to keep moving, so closing it
should stop the car, not abandon it running unattended.
"""

import argparse
import curses
import math
import sys

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
except ImportError:
    print(
        "error: rclpy not found.  source /opt/ros/<distro>/setup.bash first",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from cfr_interfaces.action import DrivePath
    from cfr_interfaces.msg import DriveCommand, PathSegment
    from std_srvs.srv import Trigger
except ImportError:
    print(
        "error: cfr_interfaces not found.  source the workspace's install/setup.bash first",
        file=sys.stderr,
    )
    sys.exit(1)


FEET_TO_METERS = 0.3048
DEG_TO_RAD = math.pi / 180.0

MIN_HEIGHT = 16
MIN_WIDTH = 60


class Segment:
    __slots__ = ("kind", "si_value", "label")

    def __init__(self, kind, si_value, label):
        self.kind = kind  # "STRAIGHT" or "TURN"
        self.si_value = si_value  # metres or radians
        self.label = label  # human-readable, for display


def build_goal(segments):
    goal = DrivePath.Goal()
    for seg in segments:
        ros_segment = PathSegment()
        if seg.kind == "STRAIGHT":
            ros_segment.type = PathSegment.STRAIGHT
            ros_segment.distance = float(seg.si_value)
        else:
            ros_segment.type = PathSegment.TURN
            ros_segment.turn_angle = float(seg.si_value)
        goal.segments.append(ros_segment)
    return goal


class PathTui:
    def __init__(
        self,
        stdscr,
        node,
        client,
        reset_odometry_client,
        action_name,
        drive_command_topic,
    ):
        self.stdscr = stdscr
        self.node = node
        self.client = client
        self.reset_odometry_client = reset_odometry_client
        self.action_name = action_name
        self.latest_drive_command = None
        self.drive_command_subscription = node.create_subscription(
            DriveCommand, drive_command_topic, self.on_drive_command, 10
        )

        self.segments = []
        self.state = "BUILDING"  # BUILDING, SENDING, EXECUTING, DONE
        self.status_message = ""
        self.final_message = None

        self.goal_handle = None
        self.goal_future = None
        self.result_future = None
        self.reset_odometry_future = None
        self.latest_feedback = None
        self.result = None

        curses.curs_set(0)
        self.colors = self._init_colors()

    def _init_colors(self):
        """Colors for distinguishing STRAIGHT/TURN segments.  Returns {} on a
        terminal without color support -- every lookup below uses .get(key, 0),
        so that degrades to no color rather than breaking."""
        if not curses.has_colors():
            return {}
        curses.start_color()
        try:
            curses.use_default_colors()
            background = -1  # terminal's own background, not forced black
        except curses.error:
            background = curses.COLOR_BLACK
        curses.init_pair(1, curses.COLOR_GREEN, background)
        curses.init_pair(2, curses.COLOR_YELLOW, background)
        curses.init_pair(3, curses.COLOR_RED, background)
        return {
            "STRAIGHT": curses.color_pair(1),
            "TURN": curses.color_pair(2),
            "ok": curses.color_pair(1) | curses.A_BOLD,
            "fail": curses.color_pair(3) | curses.A_BOLD,
        }

    # -- rclpy plumbing ------------------------------------------------------

    def spin(self):
        # A single spin_once only services one ready callback; loop a few
        # times so feedback and a result arriving in the same tick don't get
        # split across two redraws.
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def send_goal(self):
        goal = build_goal(self.segments)
        self.state = "SENDING"
        self.status_message = "sending goal..."
        self.goal_future = self.client.send_goal_async(
            goal, feedback_callback=self.on_feedback
        )

    def on_feedback(self, feedback_msg):
        self.latest_feedback = feedback_msg.feedback

    def on_drive_command(self, message):
        self.latest_drive_command = message

    def poll_futures(self):
        if self.reset_odometry_future is not None and self.reset_odometry_future.done():
            response = self.reset_odometry_future.result()
            self.status_message = (
                response.message if response.success else f"error: {response.message}"
            )
            self.reset_odometry_future = None

        if (
            self.state == "SENDING"
            and self.goal_future is not None
            and self.goal_future.done()
        ):
            self.goal_handle = self.goal_future.result()
            if not self.goal_handle.accepted:
                self.state = "DONE"
                self.result = None
                self.status_message = (
                    "REJECTED (a goal may already be running, or odometry is stale)"
                )
            else:
                self.result_future = self.goal_handle.get_result_async()
                self.state = "EXECUTING"
                self.status_message = ""

        if (
            self.state == "EXECUTING"
            and self.result_future is not None
            and self.result_future.done()
        ):
            self.result = self.result_future.result().result
            self.state = "DONE"

    def cancel_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    def reset_odometry(self):
        if self.reset_odometry_future is not None:
            return
        if not self.reset_odometry_client.service_is_ready():
            self.flash_error("odometry reset service is not available")
            return
        self.status_message = "resetting local odometry..."
        self.reset_odometry_future = self.reset_odometry_client.call_async(
            Trigger.Request()
        )

    def drain_cancel(self, ticks=20):
        """Spin briefly so a just-issued cancel has a chance to resolve before
        the process exits."""
        for _ in range(ticks):
            self.spin()
            if self.result_future is not None and self.result_future.done():
                self.result = self.result_future.result().result
                return

    # -- input -----------------------------------------------------------------

    def prompt(self, label):
        """Blocking single-line text prompt on the message row.  Returns the
        raw string ('' if the user just hit Enter)."""
        height, _width = self.stdscr.getmaxyx()
        row = height - 2
        self.stdscr.move(row, 0)
        self.stdscr.clrtoeol()
        self.stdscr.addstr(row, 0, label)
        self.stdscr.refresh()

        # getstr() needs a fully blocking read; the cooperative getch()
        # timeout used elsewhere would otherwise truncate it.
        self.stdscr.timeout(-1)
        curses.echo()
        curses.curs_set(1)
        try:
            raw = self.stdscr.getstr(row, len(label), 32).decode("utf-8").strip()
        finally:
            curses.noecho()
            curses.curs_set(0)
            self.stdscr.timeout(100)
        return raw

    def add_straight(self):
        raw = self.prompt("Distance in feet (+ forward, - reverse): ")
        if not raw:
            return
        try:
            feet = float(raw)
        except ValueError:
            self.flash_error(f"not a number: {raw!r}")
            return
        if feet == 0.0:
            self.flash_error("distance must be nonzero")
            return
        meters = feet * FEET_TO_METERS
        self.segments.append(
            Segment("STRAIGHT", meters, f"STRAIGHT  {feet:+.2f} ft   ({meters:+.3f} m)")
        )
        self.status_message = ""

    def add_turn(self):
        raw = self.prompt("Turn angle in degrees (+ left, - right): ")
        if not raw:
            return
        try:
            degrees = float(raw)
        except ValueError:
            self.flash_error(f"not a number: {raw!r}")
            return
        if degrees == 0.0:
            self.flash_error("turn angle must be nonzero")
            return
        if abs(degrees) > 180.0:
            self.flash_error(
                "turn angle must be within +/-180 deg; split into multiple segments"
            )
            return
        radians = degrees * DEG_TO_RAD
        direction = "left" if degrees > 0 else "right"
        label = f"TURN      {abs(degrees):.1f} deg {direction:<5s}({radians:+.3f} rad)"
        self.segments.append(Segment("TURN", radians, label))
        self.status_message = ""

    def flash_error(self, message):
        self.status_message = f"error: {message}"

    def handle_key_building(self, key):
        if key in (ord("s"), ord("S")):
            self.add_straight()
        elif key in (ord("t"), ord("T")):
            self.add_turn()
        elif key in (ord("d"), ord("D")):
            if self.segments:
                self.segments.pop()
                self.status_message = ""
            else:
                self.flash_error("no segments to delete")
        elif key in (ord("c"), ord("C")):
            self.segments = []
            self.status_message = ""
        elif key in (ord("r"), ord("R")):
            self.reset_odometry()
        elif key in (ord("g"), ord("G"), 10, 13):  # g, Enter
            if not self.segments:
                self.flash_error("add at least one segment first")
            elif not self.client.server_is_ready():
                self.flash_error(
                    "action server not available -- is path_follower_node running?"
                )
            else:
                self.status_message = ""
                self.send_goal()
        elif key in (ord("q"), ord("Q")):
            return "quit"
        return None

    def handle_key_executing(self, key):
        if key in (ord("x"), ord("X")):
            self.cancel_goal()
            self.status_message = "canceling..."
        elif key in (ord("q"), ord("Q")):
            self.cancel_goal()
            return "quit"
        return None

    def handle_key_done(self, key):
        if key != -1:
            self.reset_after_done()
        return None

    def reset_after_done(self):
        # Record the outcome before clearing it: curses uses the alternate
        # screen buffer, so the DONE screen leaves nothing in scrollback once
        # the TUI exits.  self.final_message is what actually gets printed
        # after curses tears down, however many segments the operator builds
        # (and quits from) afterward.
        if self.result is not None:
            outcome = "SUCCESS" if self.result.success else "FAILED"
            self.final_message = f"last result: {outcome} - {self.result.message}"
        else:
            self.final_message = "last result: REJECTED"

        # Segments are kept so a failed/canceled run can be resent as-is;
        # press 'c' to start over.
        self.state = "BUILDING"
        self.status_message = ""
        self.goal_handle = None
        self.goal_future = None
        self.result_future = None
        self.latest_feedback = None
        self.result = None

    # -- drawing -----------------------------------------------------------------

    def draw(self):
        try:
            self._draw_impl()
        except curses.error:
            pass  # terminal too small for this frame; skip rather than crash

    def _draw_execution_telemetry(self, feedback):
        self.stdscr.addstr(
            7,
            2,
            "current  x={:.3f} y={:.3f} yaw={:.1f} deg".format(
                feedback.current_x,
                feedback.current_y,
                math.degrees(feedback.current_yaw),
            ),
        )
        if feedback.desired_position_valid:
            desired_position = "x={:.3f} y={:.3f}".format(
                feedback.desired_x, feedback.desired_y
            )
            error_position = "x={:.3f} y={:.3f}".format(
                feedback.error_x, feedback.error_y
            )
        else:
            desired_position = "x=n/a y=n/a"
            error_position = "x=n/a y=n/a"
        self.stdscr.addstr(
            8,
            2,
            "desired  {} yaw={:.1f} deg".format(
                desired_position, math.degrees(feedback.desired_yaw)
            ),
        )
        self.stdscr.addstr(
            9,
            2,
            "error    {} yaw={:.1f} deg".format(
                error_position, math.degrees(feedback.error_yaw)
            ),
        )
        commands = "velocity={:+.3f} m/s yaw={:+.3f} rad/s".format(
            feedback.commanded_linear_x, feedback.commanded_angular_z
        )
        if self.latest_drive_command is not None:
            commands += "  throttle={:+.3f} steering={:+.3f}".format(
                self.latest_drive_command.throttle,
                self.latest_drive_command.steering,
            )
        self.stdscr.addstr(10, 2, commands)

    def _draw_impl(self):
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()

        connected = self.client.server_is_ready()
        status = "[CONNECTED]" if connected else "[waiting for action server...]"
        header = "CfR Path Builder"
        pad = max(1, width - len(header) - len(status) - 1)
        self.stdscr.addstr(0, 0, header + " " * pad + status)
        self.stdscr.addstr(1, 0, "-" * min(width - 1, 70))
        self.stdscr.addstr(2, 0, f"action: {self.action_name}")

        if self.state == "BUILDING":
            self.stdscr.addstr(4, 0, "Segments:")
            if not self.segments:
                self.stdscr.addstr(5, 2, "(none yet)")
            for i, seg in enumerate(self.segments):
                color = self.colors.get(seg.kind, 0)
                self.stdscr.addstr(5 + i, 2, f"{i + 1}. {seg.label}", color)
            footer_row = height - 4
            self.stdscr.addstr(footer_row, 0, "[s] add ")
            self.stdscr.addstr("straight", self.colors.get("STRAIGHT", 0))
            self.stdscr.addstr("   [t] add ")
            self.stdscr.addstr("turn", self.colors.get("TURN", 0))
            self.stdscr.addstr(
                "   [d] delete last   [c] clear all   [r] reset odometry"
            )
            self.stdscr.addstr(
                footer_row + 1, 0, "[g]/[Enter] SEND and execute        [q] quit"
            )

        elif self.state in ("SENDING", "EXECUTING"):
            self.stdscr.addstr(4, 0, self.state + "...")
            if self.latest_feedback is not None:
                fb = self.latest_feedback
                bar_width = 20
                filled = int(round(fb.segment_progress * bar_width))
                bar = "#" * filled + "-" * (bar_width - filled)
                color = 0
                if 0 <= fb.current_segment < len(self.segments):
                    color = self.colors.get(self.segments[fb.current_segment].kind, 0)
                self.stdscr.addstr(
                    5,
                    2,
                    f"segment {fb.current_segment + 1}/{fb.total_segments}  "
                    f"[{bar}] {fb.segment_progress * 100:5.1f}%",
                    color,
                )
                self._draw_execution_telemetry(fb)
            footer_row = height - 4
            self.stdscr.addstr(
                footer_row, 0, "[x] cancel        [q] quit (cancels first)"
            )

        elif self.state == "DONE":
            if self.result is not None:
                outcome = "SUCCESS" if self.result.success else "FAILED"
                color = self.colors.get("ok" if self.result.success else "fail", 0)
                self.stdscr.addstr(
                    4, 0, f"result: {outcome} - {self.result.message}", color
                )
            else:
                self.stdscr.addstr(4, 0, "result: REJECTED", self.colors.get("fail", 0))
            self.stdscr.addstr(6, 0, "press any key to build another path")

        if self.status_message:
            self.stdscr.addstr(height - 2, 0, self.status_message[: width - 1])

        self.stdscr.refresh()

    # -- main loop -----------------------------------------------------------------

    def run(self):
        self.stdscr.timeout(100)
        try:
            while True:
                self.spin()
                self.poll_futures()
                self.draw()

                key = self.stdscr.getch()
                action = None
                if self.state == "BUILDING":
                    action = self.handle_key_building(key)
                elif self.state in ("SENDING", "EXECUTING"):
                    action = self.handle_key_executing(key)
                elif self.state == "DONE":
                    action = self.handle_key_done(key)

                if action == "quit":
                    if self.state == "EXECUTING":
                        self.drain_cancel()
                    return self.final_message
        except KeyboardInterrupt:
            if self.state == "EXECUTING":
                self.cancel_goal()
                self.drain_cancel()
                if self.result is not None:
                    outcome = "SUCCESS" if self.result.success else "FAILED"
                    return f"{outcome}: {self.result.message} (interrupted)"
                return "interrupted, cancel may not have been confirmed -- check /path_follower/status"
            return self.final_message


def main(stdscr, action_name, reset_odometry_service, drive_command_topic):
    height, width = stdscr.getmaxyx()
    if height < MIN_HEIGHT or width < MIN_WIDTH:
        return f"terminal too small ({width}x{height}); need at least {MIN_WIDTH}x{MIN_HEIGHT}"

    rclpy.init(args=sys.argv)
    node = Node("path_tui")
    client = ActionClient(node, DrivePath, action_name)
    reset_odometry_client = node.create_client(Trigger, reset_odometry_service)
    try:
        tui = PathTui(
            stdscr,
            node,
            client,
            reset_odometry_client,
            action_name,
            drive_command_topic,
        )
        return tui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--action",
        default="/path_follower/drive_path",
        help="DrivePath action name (default: %(default)s)",
    )
    parser.add_argument(
        "--reset-odometry-service",
        default="/path_follower/reset_odometry",
        help="Trigger service that resets the local odometry frame (default: %(default)s)",
    )
    parser.add_argument(
        "--drive-command-topic",
        default="/drive_cmd",
        help="DriveCommand topic shown during execution (default: %(default)s)",
    )
    cli_args, _ros_args = parser.parse_known_args()

    result_message = curses.wrapper(
        main,
        cli_args.action,
        cli_args.reset_odometry_service,
        cli_args.drive_command_topic,
    )
    if result_message:
        print(result_message)
