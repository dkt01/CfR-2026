"""Run Lab: pull runs off the Orin, analyse them, watch them back in Rerun.

    web/run-lab/run.sh          # builds the UI if needed, serves on :8765

Everything is local.  Runs live in <repo>/runs (the directory sync_runs.sh
already fills and .gitignore already excludes), or $CFR_RUNS_LOCAL.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import course as course_mod  # noqa: E402
import orin  # noqa: E402
from jobs import JobRunner  # noqa: E402

REPO = HERE.parents[2]
RUNS = Path(os.environ.get("CFR_RUNS_LOCAL", REPO / "runs")).expanduser()
RUNS.mkdir(parents=True, exist_ok=True)
SETTINGS = RUNS / ".runlab.json"
DIST = HERE.parent / "frontend" / "dist"

app = FastAPI(title="CfR Run Lab")


@app.middleware("http")
async def revalidate(request, call_next):
    """Re-processing rewrites a run's analysis in place, under the same URL.
    Without Cache-Control the browser guesses a freshness period from
    Last-Modified and keeps serving the old summary and recording.rrd without
    asking.  no-cache makes it revalidate every time: an unchanged file is a
    cheap 304 on its ETag, a changed one is fetched."""
    response = await call_next(request)
    # Everything but the build's content-hashed assets: the API, and the page
    # itself -- a stale index.html loads a stale app bundle.
    if not request.url.path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response
jobs = JobRunner()


def settings():
    base = {"host": orin.DEFAULT_HOST, "remote": orin.DEFAULT_REMOTE}
    try:
        base.update(json.loads(SETTINGS.read_text()))
    except (OSError, ValueError):
        pass
    return base


def run_dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise HTTPException(400, "bad run name")
    path = RUNS / name
    if not path.is_dir():
        raise HTTPException(404, f"no run {name}")
    return path


def analysis_file(name, filename):
    path = run_dir(name) / "analysis" / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found; process the run first")
    return path


# ------------------------------------------------------------------ general


@app.get("/api/health")
def health():
    return {
        "runs_dir": str(RUNS),
        "version": analyze.VERSION,
        "settings": settings(),
    }


@app.post("/api/settings")
def save_settings(body: dict = Body(...)):
    current = settings()
    for key in ("host", "remote"):
        if key in body and isinstance(body[key], str) and body[key].strip():
            current[key] = body[key].strip()
    SETTINGS.write_text(json.dumps(current, indent=1))
    return current


@app.get("/api/jobs")
def list_jobs():
    return jobs.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.as_dict()


# --------------------------------------------------------------------- orin


@app.get("/api/orin")
def orin_status():
    s = settings()
    status = orin.status(s["host"], s["remote"])
    local = {p.name for p in RUNS.iterdir() if p.is_dir()}
    for run in status.get("runs", []):
        run["local"] = run["name"] in local
        marker = RUNS / run["name"] / ".pulled.json"
        if marker.exists():
            try:
                run["pulled"] = json.loads(marker.read_text())
            except ValueError:
                pass
    return status


@app.post("/api/orin/pull")
def orin_pull(body: dict = Body(...)):
    s = settings()
    name = body.get("run", "")
    process_after = bool(body.get("process", True))

    def work(job):
        result = orin.pull(name, RUNS, job, s["host"], s["remote"])
        if process_after and analyze_ready(RUNS / name):
            job.update(0.98, "queued for processing")
            submit_process(name)
        return result

    return jobs.submit("pull", name, work).as_dict()


@app.post("/api/orin/delete")
def orin_delete(body: dict = Body(...)):
    s = settings()
    try:
        return orin.delete(body.get("run", ""), RUNS, s["host"], s["remote"])
    except (RuntimeError, ValueError) as error:
        raise HTTPException(409, str(error)) from error


# --------------------------------------------------------------------- runs


def analyze_ready(path: Path):
    return (
        (path / "bag").is_dir()
        or any(path.glob("*.mcap"))
        or (path / "metadata.yaml").exists()
        and (path / "bag").exists()
    )


def run_entry(path: Path):
    meta = {}
    if (path / "metadata.yaml").exists():
        try:
            meta = yaml.safe_load((path / "metadata.yaml").read_text()) or {}
        except yaml.YAMLError:
            meta = {}
    entry = {
        "name": path.name,
        "kind": meta.get("kind")
        or ("characterization" if meta.get("profile") else "drive"),
        "label": meta.get("label") or meta.get("profile") or path.name,
        "started_utc": meta.get("started_utc"),
        "driver": meta.get("driver"),
        "speed_scale": meta.get("speed_scale"),
        "has_bag": (path / "bag").is_dir() or any(path.glob("*.mcap")),
        "recording": (path / "RECORDING").exists(),
        "processed": False,
    }
    summary_path = path / "analysis" / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            entry["processed"] = True
            entry["stale"] = summary.get("meta", {}).get("version") != analyze.VERSION
            entry["kpis"] = summary.get("kpis")
            entry["simulation"] = summary.get("meta", {}).get("simulation")
            entry["verdict_counts"] = {
                k: len(v) for k, v in summary.get("verdicts", {}).items()
            }
        except (OSError, ValueError):
            pass
    try:
        entry["pulled"] = json.loads((path / ".pulled.json").read_text())
    except (OSError, ValueError):
        pass
    job = jobs.active("process", path.name)
    if job:
        entry["job"] = job.as_dict()
    return entry


@app.get("/api/runs")
def list_runs():
    runs = []
    for path in sorted(RUNS.iterdir(), reverse=True):
        if path.is_dir() and not path.name.startswith("."):
            runs.append(run_entry(path))
    return runs


@app.get("/api/runs/{name}")
def get_run(name: str):
    return run_entry(run_dir(name))


def submit_process(name, clouds=True, images=True):
    path = RUNS / name

    def work(job):
        summary = analyze.process(path, job.update, clouds=clouds, images=images)
        return {"kpis": summary["kpis"]}

    return jobs.submit("process", name, work)


@app.post("/api/runs/{name}/process")
def process_run(name: str, body: dict = Body(default={})):
    run_dir(name)
    return submit_process(
        name, clouds=body.get("clouds", True), images=body.get("images", True)
    ).as_dict()


# Folders a run may be linked in from: this laptop's home directory (where a
# teammate's copy or a custom sync_runs.sh --local usually lands) and the
# usual removable-media mount points (a USB stick). Anything outside these
# is refused before it ever touches the filesystem, so a request can't be
# used to link in an arbitrary path on the machine.
IMPORT_PREFIXES = tuple(
    str(Path(p)) + os.sep
    for p in (os.path.expanduser("~"), "/media", "/mnt", "/run/media", "/Volumes")
)


@app.post("/api/runs/import")
def import_run(body: dict = Body(...)):
    """Link an existing run folder (a USB stick, a sync_runs.sh pull) in."""
    raw = (body.get("path") or "").strip()
    if not raw:
        raise HTTPException(400, "path required")
    normalized = os.path.normpath(os.path.expanduser(raw))
    # Keep this a direct normpath-then-startswith check: CodeQL only
    # recognises that exact shape as a path-injection sanitizer.
    if not normalized.startswith(IMPORT_PREFIXES):
        raise HTTPException(
            400, f"{normalized} is outside the allowed import locations"
        )
    source = Path(normalized).resolve()
    if not source.is_dir():
        raise HTTPException(400, f"{source} is not a directory")
    target = RUNS / source.name
    if target.exists():
        raise HTTPException(409, f"{source.name} already exists")
    target.symlink_to(source, target_is_directory=True)
    return run_entry(target)


@app.delete("/api/runs/{name}")
def delete_run(name: str, analysis_only: bool = False):
    path = run_dir(name)
    if analysis_only:
        shutil.rmtree(path / "analysis", ignore_errors=True)
    elif path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)
    return {"deleted": name, "analysis_only": analysis_only}


@app.get("/api/runs/{name}/summary")
def summary(name: str):
    return FileResponse(
        analysis_file(name, "summary.json"), media_type="application/json"
    )


@app.get("/api/runs/{name}/series")
def series(name: str):
    return FileResponse(
        analysis_file(name, "series.json"), media_type="application/json"
    )


@app.get("/api/runs/{name}/course")
def course(name: str):
    path = run_dir(name) / "analysis" / "course.json"
    if path.exists():
        return FileResponse(path, media_type="application/json")
    config = course_mod.load_config(run_dir(name))
    return {**course_mod.geometry(config), "sections": course_mod.zones(config)}


@app.get("/api/course")
def default_course():
    config = course_mod.load_config()
    return {**course_mod.geometry(config), "sections": course_mod.zones(config)}


@app.get("/api/runs/{name}/recording.rrd")
def recording(name: str):
    """The run for the Rerun viewer (embedded in the Replay page)."""
    return FileResponse(
        analysis_file(name, "recording.rrd"),
        media_type="application/octet-stream",
        filename=f"{name}.rrd",
        content_disposition_type="inline",
    )


@app.get("/api/runs/{name}/blueprint.rbl")
def recording_blueprint(name: str, follow: bool = False, view: str = "replay"):
    """The recording's layout, opened on the Course view or (follow=1) the
    view that rides with the car.  The viewer applies a blueprint it opens
    to the recording with the same application id."""
    import tempfile

    import rerun_export

    summary = json.loads(analysis_file(name, "summary.json").read_text())
    if view == "pose2d":
        # Framed on the pose itself, from the series the analysis already has.
        series = json.loads(analysis_file(name, "series.json").read_text())
        xs = [v for v in series["columns"].get("x_map", []) if v is not None]
        ys = [v for v in series["columns"].get("y_map", []) if v is not None]
        extent = [min(xs), max(xs), min(ys), max(ys)] if xs and ys else None
        bp = rerun_export.pose2d_blueprint(extent)
    else:
        bp = rerun_export.blueprint(
            (summary.get("perception") or {}).get("camera_size"), follow=follow
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layout.rbl"
        bp.save(rerun_export.APP_ID, path)
        data = path.read_bytes()
    return Response(data, media_type="application/octet-stream")


# One native Rerun viewer at a time, owned by the server.
_viewer = {"proc": None}


@app.post("/api/runs/{name}/rerun")
def open_native_rerun(name: str, body: dict = Body(default={})):
    """Open the native Rerun viewer on this machine: the analysed recording,
    or (which="bag") the raw MCAP bag, which Rerun reads directly."""
    path = run_dir(name)
    if body.get("which") == "bag":
        files = sorted((path / "bag").glob("*.mcap"))
        if not files:
            raise HTTPException(404, "no .mcap files in this run's bag")
        target = [str(f) for f in files]
    else:
        target = [str(analysis_file(name, "recording.rrd"))]
    rerun_bin = Path(sys.executable).with_name("rerun")
    if not rerun_bin.exists():
        raise HTTPException(
            409, "the rerun viewer is not installed in the Run Lab venv"
        )
    old = _viewer["proc"]
    if old is not None and old.poll() is None:
        old.terminate()
    _viewer["proc"] = subprocess.Popen(
        [str(rerun_bin), *target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"opened": target}


# -------------------------------------------------------------------- files

GROUPS = [
    ("Bag", lambda rel: rel.parts[0] == "bag"),
    ("Policy", lambda rel: rel.parts[0] == "policy"),
    ("ZED", lambda rel: rel.parts[0] == "zed"),
    ("Parameters", lambda rel: rel.parts[0] == "params"),
    ("Logs", lambda rel: rel.parts[0] == "logs" or rel.suffix == ".log"),
    ("Analysis", lambda rel: rel.parts[0] == "analysis"),
    ("Run", lambda rel: True),
]


@app.get("/api/runs/{name}/files")
def files(name: str):
    root = run_dir(name)
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        group = next(g for g, test in GROUPS if test(rel))
        out.append({"path": str(rel), "bytes": path.stat().st_size, "group": group})
    return out


@app.get("/api/runs/{name}/file")
def download(name: str, path: str):
    root = run_dir(name).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target, filename=target.name)


@app.get("/api/runs/{name}/archive")
def archive(name: str, analysis: bool = False):
    """The whole run as one .tar, streamed -- bags can be gigabytes."""
    run_dir(name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise HTTPException(400, "bad run name")
    cmd = ["tar", "-C", str(RUNS), "-chf", "-"]
    if not analysis:
        cmd += ["--exclude", f"{name}/analysis"]
    # "--" stops tar from ever reading `name` as an option, even though the
    # pattern above already rules out anything but [A-Za-z0-9_.-].
    cmd += ["--", name]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def stream():
        try:
            while chunk := proc.stdout.read(1 << 20):
                yield chunk
        finally:
            proc.kill()

    return StreamingResponse(
        stream(),
        media_type="application/x-tar",
        headers={"Content-Disposition": f'attachment; filename="{name}.tar"'},
    )


@app.get("/api/runs/{name}/report.md")
def report(name: str):
    s = json.loads(analysis_file(name, "summary.json").read_text())
    return Response(
        markdown_report(name, s),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{name}_report.md"'},
    )


def markdown_report(name, s):
    lines = [f"# Run {name}", ""]
    meta = s.get("meta", {}).get("metadata", {})
    for key in (
        "label",
        "started_utc",
        "driver",
        "speed_scale",
        "git_sha",
        "surface",
        "notes",
    ):
        if meta.get(key):
            lines.append(f"- **{key}**: {meta[key]}")
    lines += ["", "## Headline", ""]
    for k in s.get("kpis", []):
        lines.append(
            f"- {k['label']}: {k.get('value')} {k.get('unit', '') or ''}".rstrip()
        )
    for title, key in (("What worked", "worked"), ("What did not", "failed")):
        lines += ["", f"## {title}", ""]
        for v in s.get("verdicts", {}).get(key, []):
            lines.append(f"- **{v['title']}** -- {v['detail']}")
    laps = s.get("laps", {}).get("laps", [])
    if laps:
        lines += [
            "",
            "## Laps",
            "",
            "| lap | time s | max m/s | min clearance m | mean abs CTE m |",
            "|---|---|---|---|---|",
        ]
        for lap in laps:
            lines.append(
                f"| {lap['lap']} | {lap['time']} | {lap['max_speed']} | {lap['min_clearance']} | {lap['mean_abs_cte']} |"
            )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- static

if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:

    @app.get("/")
    def no_ui():
        return JSONResponse(
            {
                "error": "UI not built: cd web/run-lab/frontend && npm ci && npm run build"
            },
            503,
        )
