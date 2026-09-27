"""FormulaTwoEnv split across worker processes.

The depth render is most of an env step, it is numpy, and numpy holds the GIL,
so threads do not help.  Each worker owns a batched FormulaTwoEnv of n/W cars
and the parent concatenates.  Workers are spawned, not forked, so they never
inherit torch's thread pool from the trainer.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(conn, config, n, seed, deterministic, random_start):
    from pathlib import Path

    import track as track_mod
    from env import FormulaTwoEnv

    root = Path(__file__).resolve().parents[2]
    env = FormulaTwoEnv(config, track_mod.build(config, root), n, seed, deterministic)
    if random_start is not None:
        env.random_start = random_start
    while True:
        cmd, arg = conn.recv()
        if cmd == "step":
            conn.send(env.step(arg))
        elif cmd == "reset":
            conn.send(env.reset())
        elif cmd == "set":
            setattr(env, arg[0], arg[1])
            conn.send(None)
        elif cmd == "get":
            conn.send(getattr(env, arg))
        elif cmd == "close":
            conn.close()
            return


class ParallelEnv:
    def __init__(
        self, config, n_envs, workers, seed=0, deterministic=False, random_start=None
    ):
        ctx = mp.get_context("spawn")
        sizes = [n_envs // workers + (i < n_envs % workers) for i in range(workers)]
        self.sizes = sizes
        self.n = n_envs
        self.conns = []
        self.procs = []
        for i, k in enumerate(sizes):
            a, b = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(b, config, k, seed * 1000 + i, deterministic, random_start),
                daemon=True,
            )
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.obs_dim = self._get("obs_dim")
        self.actor_dim = self._get("actor_dim")
        self.map_dim = self._get("map_dim")
        self.act_dim = 2

    def _get(self, name):
        self.conns[0].send(("get", name))
        return self.conns[0].recv()

    def set(self, name, value):
        for c in self.conns:
            c.send(("set", (name, value)))
        for c in self.conns:
            c.recv()

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        start = 0
        for c, k in zip(self.conns, self.sizes):
            c.send(("step", actions[start : start + k]))
            start += k
        parts = [c.recv() for c in self.conns]
        obs = np.concatenate([p[0] for p in parts])
        rew = np.concatenate([p[1] for p in parts])
        term = np.concatenate([p[2] for p in parts])
        trunc = np.concatenate([p[3] for p in parts])
        info = [i for p in parts for i in p[4]]
        return obs, rew, term, trunc, info

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)
