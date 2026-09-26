#!/usr/bin/env python3
"""Drive a policy.npz in its numpy training sim and save one episode's track.

Runs on the host, with rl/obstacleRacer/.venv's python (numba for the
obstacle course; the Speed Course needs only numpy and yaml):

    rollout.py --course obstacle --policy rl/obstacleRacer/runs/v6/policy.npz \
        --seed 104 --episodes 24 --pick fastest --out /tmp/track.npz
    rollout.py --course speed --policy rl/formulaOne/bestModel/v12/policy.npz \
        --out /tmp/track.npz

The track is what capture.py replays: t, x, y, z, yaw, and pitch/roll in
Gazebo's convention (pitch nose-DOWN positive -- plant.py stores nose-up
positive, so it is negated here), plus v, vcmd, steer and a per-step progress
count (hoops threaded, or laps completed).  A sidecar .json holds the outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]


def jsonable(d):
    return {
        k: v
        for k, v in d.items()
        if isinstance(v, (int, float, str, bool, list)) or v is None
    }


def obstacle(args):
    here = REPO / "rl/obstacleRacer"
    sys.path.insert(0, str(here))
    import course_model
    import observation as O
    import plant as P
    from policy import NumpyPolicy
    from train import EvalEnv

    cfg = yaml.safe_load(args.config.read_text())
    model = course_model.CourseModel([args.seed])
    n = args.episodes
    env = EvalEnv(cfg, model, n, np.array([0]), seed=args.rng, start_box_only=True)
    env.layout_ids_cycle = np.zeros(n, int)
    net = NumpyPolicy.load(args.policy)
    if hasattr(net, "reset"):
        net.reset(n)  # a recurrent policy starts its memory with the run
    obs = env.reset()
    rows = [[] for _ in range(n)]
    done = np.zeros(n, bool)
    results = [None] * n
    for _ in range(int(cfg["env"]["episode_s"] * cfg["env"]["control_hz"]) + 5):
        a = np.clip(net.act(obs), -1, 1)
        steer, vcmd = O.action_to_command(a, env.prior.copy(), cfg)
        obs, _, _, _, info = env.step(a)
        st = env.plant.state
        for i in range(n):
            if done[i]:
                continue
            if info[i]:
                results[i] = info[i]
                done[i] = True
                continue
            rows[i].append(
                [
                    env.t[i],
                    st[i, P.S_X],
                    st[i, P.S_Y],
                    st[i, P.S_Z],
                    st[i, P.S_YAW],
                    -st[i, P.S_PITCH],
                    st[i, P.S_ROLL],
                    st[i, P.S_V],
                    vcmd[i],
                    steer[i],
                    int((env.hoop_state[i] == 1).sum()),
                ]
            )
        if done.all():
            break
    summaries = []
    for i, r in enumerate(results):
        if r is None:
            continue
        ok = r["outcome"] == "finish"
        summaries.append(
            dict(
                index=i,
                finished=ok,
                outcome=r["outcome"],
                zone=r["zone"],
                time=r["time"],
                hoops=r["hoops"],
            )
        )
    return rows, summaries, "hoops", 3, float(env.dt)


def speed(args):
    here = REPO / "rl/formulaOne"
    sys.path.insert(0, str(here))
    import track as track_mod
    from env import FormulaOneEnv
    from policy import NumpyPolicy

    cfg = yaml.safe_load(args.config.read_text())
    trk = track_mod.build(cfg, REPO)
    net = NumpyPolicy.load(args.policy)
    laps = int(cfg["env"]["laps"])
    rows, summaries = [], []
    for k in range(args.episodes):
        env = FormulaOneEnv(cfg, trk, 1, args.rng + k, deterministic=True)
        env.random_start = False
        obs = env.reset()
        run = []
        info = {}
        for _ in range(20000):
            s = env.snapshot()
            run.append(
                [
                    float(s["elapsed"][0]),
                    float(s["x"][0]),
                    float(s["y"][0]),
                    0.0,
                    float(s["yaw"][0]),
                    0.0,
                    0.0,
                    float(s["speed"][0]),
                    float(s["v_cap"][0]),
                    float(env.last_steer[0]),
                    int(env.laps_done[0]),
                ]
            )
            obs, _, term, trunc, inf = env.step(net.act(obs))
            if term[0] or trunc[0]:
                info = inf[0]
                break
        rows.append(run)
        ok = bool(info.get("finished"))
        summaries.append(
            dict(
                index=k,
                finished=ok,
                outcome="finish"
                if ok
                else (
                    "crash"
                    if info.get("crashed")
                    else "stall"
                    if info.get("stalled")
                    else "timeout"
                ),
                time=float(info.get("race_time", run[-1][0]))
                if ok
                else float(run[-1][0]),
                laps=int(info.get("laps", 0)),
                best_lap=info.get("best_lap"),
                min_clearance=info.get("min_clearance"),
            )
        )
        if ok and args.pick == "first":
            break
    return rows, summaries, "lap", laps, float(1.0 / cfg["env"]["control_hz"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--course", choices=["obstacle", "speed"], required=True)
    ap.add_argument("--policy", type=Path, required=True, help="exported policy.npz")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="default: config.yaml beside the policy",
    )
    ap.add_argument("--seed", type=int, default=104, help="obstacle layout seed")
    ap.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="obstacle: parallel starts (24); speed: seeds (1)",
    )
    ap.add_argument(
        "--pick",
        choices=["fastest", "first", "median", "any"],
        default="fastest",
        help="which finishing episode; 'any' takes the longest run when none finish",
    )
    ap.add_argument("--rng", type=int, default=20260926)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.config = args.config or args.policy.with_name("config.yaml")
    if args.episodes is None:
        args.episodes = 24 if args.course == "obstacle" else 1

    rows, summaries, label, total, dt = (
        obstacle if args.course == "obstacle" else speed
    )(args)
    for s in summaries:
        print(json.dumps(s))
    fin = [s for s in summaries if s["finished"]]
    print(f"finished {len(fin)} of {len(summaries)}")
    if fin:
        fin.sort(key=lambda s: s["time"])
        best = {
            "fastest": fin[0],
            "first": min(fin, key=lambda s: s["index"]),
            "median": fin[len(fin) // 2],
        }.get(args.pick, fin[0])
    elif args.pick == "any":
        best = max(summaries, key=lambda s: len(rows[s["index"]]))
        print("no finish; taking the longest run", file=sys.stderr)
    else:
        print(
            "no episode finished; rerun with more --episodes, another --seed, or --pick any",
            file=sys.stderr,
        )
        return 2
    track = np.array(rows[best["index"]], float)
    np.savez(
        args.out,
        t=track[:, 0] - track[0, 0],
        pose=track[:, 1:7],  # x y z yaw pitch(nose-down +) roll
        v=track[:, 7],
        vcmd=track[:, 8],
        steer=track[:, 9],
        progress=track[:, 10].astype(int),
    )
    meta = dict(
        course=args.course,
        policy=str(args.policy),
        seed=args.seed if args.course == "obstacle" else None,
        dt=dt,
        progress_label=label,
        progress_total=total,
        vcmd_label="cmd v" if args.course == "obstacle" else "cap",
        finished_of=[len(fin), len(summaries)],
        **jsonable(best),
    )
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=1))
    print(
        f"saved episode {best['index']} ({best['outcome']}, {best['time']:.1f} s, {len(track)} steps) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
