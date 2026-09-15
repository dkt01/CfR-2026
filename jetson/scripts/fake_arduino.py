#!/usr/bin/env python3
"""Fake Arduino for exercising the Jetson software stack with no hardware
attached.

Opens a pseudo-terminal and symlinks it to a stable path, then speaks just
enough of the onboard protocol (see ../README.md "Wire format") to unblock
arduino_bridge_node's AUTO_ARMED -> AUTO_ACTIVE handshake: once it sees a
Jetson frame with auto_ready=1, steering centered and a zero speed target, it
reports mode=AUTO_ACTIVE and stays there.

Speed is a first order lag toward the commanded spur RPM, standing in for the
real firmware's closed loop controller.  Gains frames are checksum validated
and their sequence number echoed, so the bridge sees them applied.

This does NOT simulate the offboard XBee/RC/E-Stop link

    python3 fake_arduino.py                          # creates /tmp/fake_arduino
    ~/software/scripts/launch.sh --device /tmp/fake_arduino --skip-checks

or just:

    ~/software/scripts/launch.sh --fake-arduino
"""

import argparse
import math
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
MAX_TARGET_RPM = 20000
GAINS_FRAME_LENGTH = 82

MODE_ESTOP = 0
MODE_AUTO_ARMED = 3
MODE_AUTO_ACTIVE = 4


def is_centered(value):
    return abs(value - CENTER) <= DEADBAND


def parse_command(line):
    """Parse a drive command, always exactly "C,b,sss,+rrrrr" (see
    protocol.cpp Serialize()).  Returns None if malformed, same as the firmware
    silently dropping a bad frame."""
    if len(line) != 14 or line[:2] != "C," or line[3] != "," or line[7] != ",":
        return None
    if line[2] not in "01" or line[8] not in "+-":
        return None
    if not (line[4:7].isdigit() and line[9:14].isdigit()):
        return None
    steering = int(line[4:7])
    target = int(line[8:14])
    if steering > 255 or abs(target) > MAX_TARGET_RPM:
        return None
    return line[2] == "1", steering, target


def parse_gains_seq(line):
    """Validate a gains frame's layout and checksum; return its sequence
    number, or None.  The gains themselves are not used by the fake."""
    if len(line) != GAINS_FRAME_LENGTH or not line.startswith("G,"):
        return None
    try:
        checksum = int(line[-2:], 16)
    except ValueError:
        return None
    if sum(line[:-2].encode("ascii", errors="replace")) & 0xFF != checksum:
        return None
    if not line[2:5].isdigit() or int(line[2:5]) > 255:
        return None
    return int(line[2:5])


def _handle_sigterm(signum, frame):
    # Python only turns SIGINT into a catchable exception by default; a plain
    # `kill $pid` (what launch.sh's cleanup trap sends) would otherwise skip
    # the finally block below and leave the symlink dangling.
    raise SystemExit(0)


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--link",
        default=DEFAULT_LINK,
        help=f"symlink to the PTY (default: {DEFAULT_LINK})",
    )
    parser.add_argument(
        "--battery",
        type=int,
        default=200,
        help="reported battery level, 0-255 (default: 200)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="time constant in seconds of the simulated speed response (default: 0.5)",
    )
    args = parser.parse_args()

    if not 0 <= args.battery <= 255:
        print("error: --battery must be 0-255", file=sys.stderr)
        return 1

    if args.tau <= 0.0:
        print("error: --tau must be positive", file=sys.stderr)
        return 1

    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)

    if os.path.islink(args.link) or os.path.exists(args.link):
        os.remove(args.link)
    os.symlink(slave_path, args.link)

    print(f"fake Arduino on {slave_path} (linked from {args.link})")
    print(
        "WARNING: no real actuators, no real E-Stop -- only simulates the Jetson-side auto-arm handshake"
    )
    print("Ctrl-C to stop")
    sys.stdout.flush()

    mode = MODE_AUTO_ARMED
    target = 0
    gains_seq = 0
    rpm = 0.0
    rx_buffer = b""
    last_status = 0.0
    last_step = time.monotonic()
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
                    raw, rx_buffer = rx_buffer.split(b"\n", 1)
                    line = raw.decode("ascii", errors="replace").rstrip("\r")
                    seq = parse_gains_seq(line)
                    if seq is not None:
                        gains_seq = seq
                        continue
                    frame = parse_command(line)
                    if frame is None:
                        continue
                    auto_ready, steering, requested = frame
                    if (
                        mode == MODE_AUTO_ARMED
                        and auto_ready
                        and is_centered(steering)
                        and requested == 0
                    ):
                        mode = MODE_AUTO_ACTIVE
                    target = requested if mode == MODE_AUTO_ACTIVE else 0

            now = time.monotonic()
            dt = now - last_step
            last_step = now
            rpm += (target - rpm) * (1.0 - math.exp(-dt / args.tau))

            if now - last_status >= period:
                last_status = now
                # estop=0, auto_arm=1, manual_start=0 -- fixed, since nothing
                # here simulates the offboard link that would normally drive
                # them.  The throttle field is a nominal feedforward, not a
                # model of anything.  Trailing comma on every field matches
                # ToJetson::serialize().
                reported = int(round(rpm))
                throttle = (
                    1500 + int(math.copysign(28 + 9.5 * abs(target) / 1000, target))
                    if target
                    else 1500
                )
                status = f"0,1,0,{mode},{args.battery},{reported},{target},{throttle},{gains_seq},\n"
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
