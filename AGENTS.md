# AGENTS.md

Guidance for AI coding agents working in CfR-2026 (autonomous RC car: Jetson
Orin + ZED camera + Arduino Uno, ROS 2 Jazzy, Gazebo Harmonic).

Read [README.md](README.md) first. It covers run modes, the RC and E-Stop
protocol, the onboard serial protocol, speed control, and simulation.

## Layout

* `jetson/` -- ROS 2 workspace for the car ([jetson/README.md](jetson/README.md))
* `rl/bale_follower/` -- PPO training for the bale-following policy
* `web/run-lab/`, `web/gzweb-viewer/` -- drive-log analysis UI and browser sim viewer
* `zed/` -- ZED camera configuration
* `docs/` -- characterization procedure and results

## Skills

Task playbooks live in `.agents/skills/<name>/SKILL.md`, each with its own
scripts. Read the matching one before doing the task by hand:

* `run-tests` -- colcon build and test suite exactly as CI runs it, in Docker
* `sim-launch` -- Gazebo speed or obstacle course plus the gzweb viewer
* `run-lab` -- record, pull and analyze drive logs
* `deploy-to-car` -- sync and build `jetson/` on the Orin
* `rl-train` -- start, watch and stop PPO training runs
* `rl-reward` -- review and change reward shaping, with a before/after probe
* `obstacle-course-regions` -- map (x, y) points to named obstacles

`.claude/skills` is a symlink to `.agents/skills`, so Claude Code reads the
same files. Add or edit skills under `.agents/skills/` only. Symlinks need
`core.symlinks=true` (and Developer Mode on Windows).

## Safety

* Launching on the real car arms actuators. Never run `launch.sh` on the Orin
  unattended or in the background; the user must be at the bench with the
  E-Stop remote in hand and confirm immediately before.
* Do not change E-Stop, arming, or serial-protocol behavior without checking
  the README protocol sections.

## Conventions

* American spelling everywhere (color, meter, center, neighbor, gray) in code,
  comments, docs and UI. Leave pre-existing identifiers such as `HELIX_CENTRE`.
* The repo relies on `core.autocrlf=input`. A CRLF working tree breaks
  anything run from a Linux mount (`#!/usr/bin/env python3\r` fails with exit
  127). When editing from Windows, write bytes (or a bare `\n` newline), and
  check for CR bytes after edits and after pre-commit runs.
* Run the sim and the test suite in Docker. WSL RoboStack `colcon test` cannot
  run this repo's `launch_testing` pytest tests.
* RL: observations must be realizable from the ZED and wheel encoder; reward
  may use privileged state. The objective is fastest laps. Judge reward changes
  by episode-level incentives, not per-step ordering tests alone.
