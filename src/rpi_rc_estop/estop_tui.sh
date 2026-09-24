#!/usr/bin/env bash
#
# Launch the laptop E-Stop / RC emulator (estop_tui.py).
#
#   src/rpi_rc_estop/estop_tui.sh
#   src/rpi_rc_estop/estop_tui.sh --reinstall      # rebuild the Python environment
#   ESTOP_SERIAL_TRACE=/tmp/xbee.log src/rpi_rc_estop/estop_tui.sh   # log raw XBee traffic
#
# The XBee adapter and the gamepad can be plugged in before or after starting:
# the TUI discovers both, and finds them again if they drop out.  It starts
# with the E-Stop SET, so starting it never releases the car.  Space sets the
# E-Stop, c clears it, q then y quits.
#
# The first run sets up .venv from requirements-tui.txt.  Later runs reinstall
# only when that file has changed since the last install.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=.venv
REQS=requirements-tui.txt
# A copy of the requirements the venv was last installed from.
INSTALLED="$VENV/.installed-$REQS"
REINSTALL=false

usage() {
    sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reinstall)
            REINSTALL=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# rich's full-screen display and readchar's key reader both need a real
# terminal; without one the E-Stop keys would silently do nothing.
if [[ ! -t 0 || ! -t 1 ]]; then
    echo "error: the E-Stop TUI needs an interactive terminal" >&2
    exit 1
fi

# A sourced ROS environment puts its own site-packages (pyserial among them)
# on PYTHONPATH.  The TUI must run on its own venv only.
unset PYTHONPATH

if [[ "$REINSTALL" == true ]]; then
    rm -rf "$VENV"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "setting up the Python environment in $VENV (once)..."
    python3 -m venv "$VENV"
fi

if ! cmp -s "$REQS" "$INSTALLED"; then
    echo "installing $REQS..."
    # pygame-ce and stock pygame both install the `pygame` package, so the
    # stock one has to come out first.  Venvs made before the switch to
    # pygame-ce still carry it.
    if grep -qi '^pygame-ce' "$REQS"; then
        "$VENV/bin/python" -m pip uninstall -q -y pygame >/dev/null 2>&1 || true
    fi
    "$VENV/bin/python" -m pip install -q -r "$REQS"
    cp "$REQS" "$INSTALLED"
fi

# Serial access is checked here, not left to the TUI: once its full-screen
# display is up, a permissions error scrolls past where nobody reads it.
shopt -s nullglob
PORTS=(/dev/ttyUSB* /dev/ttyACM*)
shopt -u nullglob
if [[ ${#PORTS[@]} -eq 0 ]]; then
    echo "no USB serial adapter yet -- plug in the XBee, the TUI picks it up when it appears"
    sleep 1
fi
for port in ${PORTS[@]+"${PORTS[@]}"}; do
    if [[ ! -r "$port" || ! -w "$port" ]]; then
        echo "error: no read/write access to $port" >&2
        echo "       sudo usermod -aG dialout $USER   # then log out and back in" >&2
        exit 1
    fi
done

# pygame prints a banner on import, which lands on top of the TUI.
export PYGAME_HIDE_SUPPORT_PROMPT=1
exec "$VENV/bin/python" estop_tui.py
