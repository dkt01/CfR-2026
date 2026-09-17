---
name: run-tests
description: Run the CfR-2026 jetson/ colcon build and test suite exactly as CI does (.github/workflows/ci.yml), in a throwaway ros:jazzy-ros-base Docker container, without touching the working tree. Use this whenever the user asks to run tests, check CI, verify a change builds/passes, or run colcon test/build for this project -- especially before opening a PR or after editing jetson/cfr_arduino_bridge or jetson/cfr_interfaces. Prefer this over reconstructing the docker run / rosdep / colcon commands by hand, and over WSL's RoboStack colcon (its pytest version can't run this repo's launch_testing-based tests).
---

# CfR-2026 test runner

Wraps the CI-equivalent test sequence (rosdep install, `build.sh --test`)
into one script, `scripts/test.sh`, run inside a throwaway
`ros:jazzy-ros-base` container -- the same image and steps as the `Build and
test` job in `.github/workflows/ci.yml`.

```bash
.claude/skills/run-tests/scripts/test.sh                    # build + test everything under jetson/
.claude/skills/run-tests/scripts/test.sh --no-test           # build only, skip colcon test
.claude/skills/run-tests/scripts/test.sh cfr_arduino_bridge   # limit to one package
```

The repo is mounted **read-only** (`-v $REPO:/repo:ro`) and the container is
`--rm`, so this never leaves build artifacts in the working tree and never
needs cleanup -- every run starts from a clean image layer with a fresh
`apt-get update`/`rosdep install`, which costs ~1-2 minutes before the actual
build starts. That's the tradeoff for exactness: it reproduces CI's
dependency resolution instead of trusting whatever happens to already be
installed somewhere.

On success this prints `Summary: N tests, 0 errors, 0 failures, 0 skipped` and
`tests passed`. On failure, `colcon test-result --verbose` (run inside
`build.sh --test`) prints which specific test case failed before the script
exits non-zero.

## Why not WSL

The repo also has a RoboStack Jazzy colcon workspace in WSL (see the
`bench-hardware-setup` memory), but its pytest 9 rejects the
`launch_testing` plugin's old hookimpl -- `ament_add_pytest_test` targets
error out there. This container-based path is the one that actually matches
what CI runs and what reviewers see.

## Notes

- Needs Docker Desktop running; if `docker info` fails, say so rather than
  retrying blindly.
- This only covers the `build-and-test` CI job (the ROS packages). CI also
  has a separate `arduino-build` job (`arduino-cli compile` for the Uno
  firmware) and a `format.yml` pre-commit check -- this skill doesn't cover
  either of those.
- If the working tree has CRLF line endings from Windows-side edits (see the
  `windows-line-endings-trap` memory), that's a working-tree problem, not
  something this script needs to work around -- it only affects files that
  get *executed* inside a Linux mount, and nothing under `jetson/` is
  shebang'd and run directly by colcon.
