#!/usr/bin/env python3
"""Fake Arduino for exercising the Jetson software stack with no hardware
attached.

Opens a pseudo-terminal and symlinks it to a stable path, then speaks just
enough of the onboard protocol (see ../README.md "Wire format") to unblock
arduino_bridge_node's AUTO_ARMED -> AUTO_ACTIVE handshake: once it sees a
Jetson frame with auto_ready=1 and both axes centered, it reports
mode=AUTO_ACTIVE and stays there.

This does NOT simulate the offboard XBee/RC/E-Stop link -- there's no code for
that in this repo yet, and the real sketch's mode state machine is gated on it
too (offboardTimedOut forces Mode::ESTOP regardless of what the Jetson sends).
This fake skips that gate entirely and starts AUTO_ARMED. Use it to exercise
message flow through arduino_bridge_node, cmd_vel_to_drive_node, and
path_follower_node -- not as a stand-in for a safety-validated bench session.

    python3 fake_arduino.py                          # creates /tmp/fake_arduino
    ~/software/scripts/launch.sh --device /tmp/fake_arduino --skip-checks

or just:

    ~/software/scripts/launch.sh --fake-arduino
"""

import argparse
import os
import pty
import select
import signal
import sys
import time

DEFAULT_LINK = "/tmp/fake_arduino"
CENTER = 127
DEADBAND = 5
STATUS_RATE_HZ = 20

MODE_ESTOP = 0
MODE_AUTO_ARMED = 3
MODE_AUTO_ACTIVE = 4


def is_centered(value):
    return abs(value - CENTER) <= DEADBAND


def parse_frame(line):
    """Parse a Jetson -> Arduino frame: auto_ready,steering,throttle (no
    trailing comma, see protocol.cpp Serialize()).  Returns None if malformed,
    same as the firmware silently dropping a bad frame."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        auto_ready = parts[0] == "1"
        steering = int(parts[1])
        throttle = int(parts[2])
    except ValueError:
        return None
    if not (0 <= steering <= 255 and 0 <= throttle <= 255):
        return None
    return auto_ready, steering, throttle


def _handle_sigterm(signum, frame):
    # Python only turns SIGINT into a catchable exception by default; a plain
    # `kill $pid` (what launch.sh's cleanup trap sends) would otherwise skip
    # the finally block below and leave the symlink dangling.
    raise SystemExit(0)


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--link", default=DEFAULT_LINK, help=f"symlink to the PTY (default: {DEFAULT_LINK})")
    parser.add_argument("--battery", type=int, default=200, help="reported battery level, 0-255 (default: 200)")
    args = parser.parse_args()

    if not 0 <= args.battery <= 255:
        print("error: --battery must be 0-255", file=sys.stderr)
        return 1

    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)

    if os.path.islink(args.link) or os.path.exists(args.link):
        os.remove(args.link)
    os.symlink(slave_path, args.link)

    print(f"fake Arduino on {slave_path} (linked from {args.link})")
    print("WARNING: no real actuators, no real E-Stop -- only simulates the Jetson-side auto-arm handshake")
    print("Ctrl-C to stop")
    sys.stdout.flush()

    mode = MODE_AUTO_ARMED
    rx_buffer = b""
    last_status = 0.0
    period = 1.0 / STATUS_RATE_HZ

    try:
        while True:
            now = time.monotonic()
            timeout = max(0.0, period - (now - last_status))
            ready, _, _ = select.select([master_fd], [], [], timeout)

            if master_fd in ready:
                try:
                    chunk = os.read(master_fd, 256)
                except OSError:
                    break  # peer closed the port
                if not chunk:
                    break
                rx_buffer += chunk
                while b"\n" in rx_buffer:
                    line, rx_buffer = rx_buffer.split(b"\n", 1)
                    frame = parse_frame(line.decode("ascii", errors="replace"))
                    if frame is None:
                        continue
                    auto_ready, steering, throttle = frame
                    if mode == MODE_AUTO_ARMED and auto_ready and is_centered(steering) and is_centered(throttle):
                        mode = MODE_AUTO_ACTIVE

            now = time.monotonic()
            if now - last_status >= period:
                last_status = now
                # estop=0, auto_arm=1, manual_start=0 -- fixed, since nothing
                # here simulates the offboard link that would normally drive
                # them.  Trailing comma on every field matches ToJetson::serialize().
                status = f"0,1,0,{mode},{args.battery},\n"
                try:
                    os.write(master_fd, status.encode("ascii"))
                except OSError:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        os.close(master_fd)
        os.close(slave_fd)
        if os.path.islink(args.link):
            os.remove(args.link)
        print("\nfake Arduino stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
