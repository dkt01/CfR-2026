"""Gazebo cars beside the numpy ones, as one batch for the trainer.

`MixedEnv` looks like ObstacleEnv to train.py (n, obs_dim, reset, step):
the first `numpy.n` cars are the numpy env's, the rest are Gazebo cars, one
per sim container, each served by `gazebo_env.py serve` over the pipe of a
`docker exec -i`.  A step sends every Gazebo car its action first, steps the
numpy batch while Gazebo runs, then collects the Gazebo replies, so a step
costs about the slower of the two.

A container that stops answering (Gazebo hung, the server died) is
restarted with its stack, and that car's episode is cut there: the trainer
sees a truncation, as at a time limit, and a fresh start.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import gazebo_wire as wire

HERE = Path(__file__).resolve().parent
BOX_DIR = "/repo/rl/obstacleRacer"
IMAGE = "unfrobotics/docker-ros2-jazzy-gz-rviz2:latest"
ROS = "source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash"


class GazeboWorker:
    """One sim container and the gazebo_env.py server in it."""

    def __init__(
        self, container, cfg, seeds, seed, log_path, rtf=1.0, reply_s=180.0, domain=41
    ):
        self.container = container
        # Containers on Docker's bridge network see each other's gz transport
        # (multicast discovery) and ROS 2 traffic: two Gazebos then serve one
        # /zed/zed_node/pose and one set_pose, and a teleport lands in the
        # other world.  Each car gets its own partition and domain.
        self.isolate = [
            "-e",
            f"ROS_DOMAIN_ID={domain}",
            "-e",
            f"GZ_PARTITION=cfr_racer_{domain}",
        ]
        self.cfg = cfg
        self.seeds = list(seeds)
        self.seed = seed
        self.log_path = Path(log_path)
        self.rtf = rtf
        self.reply_s = reply_s
        self.proc = None
        self.replies = None
        self.last_obs = None
        self.restarts = 0
        self.dims = None

    # ------------------------------------------------------------ process

    def _docker(self, *args, timeout=600):
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout
        )

    def start(self):
        self.stop()
        r = self._docker("restart", self.container, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(f"docker restart {self.container}: {r.stderr.strip()}")
        r = self._docker(
            "exec",
            *self.isolate,
            self.container,
            "bash",
            "-c",
            f"tr -d '\\r' < {BOX_DIR}/gazebo_stack.sh > /tmp/gazebo_stack.sh && "
            f"CFR_REPO=/repo bash /tmp/gazebo_stack.sh --rtf {self.rtf}",
            timeout=400,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"{self.container} stack: {r.stdout[-800:]}{r.stderr[-800:]}"
            )
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                *self.isolate,
                self.container,
                "bash",
                "-c",
                f"{ROS} && cd {BOX_DIR} && exec python3 gazebo_env.py serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            bufsize=0,
        )
        self.replies = queue.Queue()
        proc, replies = self.proc, self.replies

        def reader():
            try:
                while True:
                    replies.put(wire.receive(proc.stdout))
            except Exception as error:  # noqa: BLE001 - EOF or a broken pipe
                replies.put(error)

        threading.Thread(target=reader, daemon=True).start()
        wire.send(self.proc.stdin, dict(cfg=self.cfg, seeds=self.seeds, seed=self.seed))
        self.dims = self._reply(timeout=900)  # the course model may bake first

    def stop(self):
        if self.proc is not None:
            try:
                wire.send(self.proc.stdin, dict(kind="close"))
            except Exception:  # noqa: BLE001
                pass
            try:
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            self.proc = None

    def _reply(self, timeout=None):
        got = self.replies.get(timeout=timeout or self.reply_s)
        if isinstance(got, Exception):
            raise got
        if isinstance(got, dict) and "error" in got:
            raise RuntimeError(got["error"].strip().splitlines()[-1])
        return got

    # ---------------------------------------------------------------- env

    def _reset_once(self):
        wire.send(self.proc.stdin, dict(kind="reset"))
        self.last_obs = self._reply()
        return self.last_obs

    def reset(self):
        try:
            return self._reset_once()
        except Exception as error:  # noqa: BLE001
            self._log(error)
            return self._restart()

    def _log(self, error):
        self.restarts += 1
        print(
            f"[gazebo] {self.container}: {type(error).__name__}: {error} -- restarting "
            f"(restart {self.restarts})",
            flush=True,
        )

    def _restart(self):
        for attempt in range(5):
            try:
                self.start()
                return self._reset_once()
            except Exception as again:  # noqa: BLE001
                print(f"[gazebo] {self.container}: restart failed: {again}", flush=True)
                time.sleep(30 * (attempt + 1))
        raise RuntimeError(f"{self.container} would not come back")

    def step_async(self, action):
        self.sent_at = time.monotonic()
        try:
            wire.send(self.proc.stdin, dict(kind="step", action=action[None, :]))
            self.send_failed = None
        except Exception as error:  # noqa: BLE001
            self.send_failed = error

    def step_wait(self):
        try:
            if self.send_failed is not None:
                raise self.send_failed
            obs, rew, term, trunc, info = self._reply()
            self.last_obs = obs
            info[0]["sim"] = "gazebo"
            return obs, rew, term, trunc, info
        except Exception as error:  # noqa: BLE001 - timeout, EOF, broken pipe
            return self._recover(error)

    def _recover(self, error):
        """Restart the container; the running episode ends as truncated."""
        self._log(error)
        last = self.last_obs
        obs = self._restart()
        info = [
            {
                "sim": "gazebo",
                "restarted": True,
                "TimeLimit.truncated": True,
                "terminal_observation": last[0].copy(),
            }
        ]
        return obs, np.zeros(1, np.float32), np.zeros(1, bool), np.ones(1, bool), info


def ensure_container(name, repo):
    """A sim container mounting this checkout, built and with numba (idempotent)."""
    have = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", name],
        capture_output=True,
        text=True,
    )
    if have.returncode != 0:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-v",
                f"{repo}:/repo",
                IMAGE,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
        )
    subprocess.run(["docker", "start", name], check=True, capture_output=True)
    setup = (
        "set -e; "
        "python3 -c 'import numba, scipy, PIL' 2>/dev/null || { "
        "(command -v pip3 >/dev/null || (apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >/dev/null)) && "
        "pip3 install --break-system-packages -q 'numba==0.60.*' scipy pillow 'numpy==1.26.4'; }; "
        "test -f ~/ros2_ws/install/setup.bash || (source /opt/ros/jazzy/setup.bash && "
        "cd /repo && ./jetson/scripts/build.sh >/tmp/build.log 2>&1)"
    )
    r = subprocess.run(
        ["docker", "exec", name, "bash", "-c", setup],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{name} setup failed: {r.stdout[-800:]}{r.stderr[-800:]}")


class MixedEnv:
    """The numpy batch, then one car per GazeboWorker."""

    def __init__(self, numpy_env, workers):
        self.numpy = numpy_env
        self.workers = workers
        self.n_np = numpy_env.n
        self.n = self.n_np + len(workers)
        self.obs_dim = numpy_env.obs_dim
        self.act_dim = numpy_env.act_dim
        self.step_wall = []

    def __getattr__(self, name):
        # zone stats, section weights, cfg: the numpy env's
        return getattr(self.__dict__["numpy"], name)

    def reset(self):
        obs = [self.numpy.reset()]
        obs += [w.reset() for w in self.workers]
        return np.concatenate(obs).astype(np.float32)

    def step(self, actions):
        actions = np.asarray(actions)
        for k, w in enumerate(self.workers):
            w.step_async(np.asarray(actions[self.n_np + k], np.float64))
        t0 = time.monotonic()
        obs, rew, term, trunc, info = self.numpy.step(actions[: self.n_np])
        t_np = time.monotonic() - t0
        parts = [w.step_wait() for w in self.workers]
        self.step_wall.append((t_np, time.monotonic() - t0))
        obs = np.concatenate([obs] + [p[0] for p in parts]).astype(np.float32)
        rew = np.concatenate([rew] + [np.asarray(p[1], np.float32) for p in parts])
        term = np.concatenate([term] + [np.asarray(p[2], bool) for p in parts])
        trunc = np.concatenate([trunc] + [np.asarray(p[3], bool) for p in parts])
        info = list(info) + [p[4][0] for p in parts]
        return obs, rew, term, trunc, info

    def close(self):
        for w in self.workers:
            w.stop()
