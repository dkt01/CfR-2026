"""Background jobs (pull, process) with progress the browser can poll."""

from __future__ import annotations

import threading
import time
import traceback
import uuid


class Job:
    def __init__(self, kind, target):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.target = target
        self.state = "queued"
        self.progress = 0.0
        self.message = ""
        self.lines = []
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None

    def update(self, progress, message):
        self.progress = max(0.0, min(1.0, float(progress)))
        self.message = message

    def log(self, line):
        self.lines.append(line)
        del self.lines[:-200]

    def as_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "state": self.state,
            "progress": round(self.progress, 3),
            "message": self.message,
            "lines": self.lines[-20:],
            "result": self.result,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
        }


class JobRunner:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def active(self, kind, target):
        with self.lock:
            for job in self.jobs.values():
                if (
                    job.kind == kind
                    and job.target == target
                    and job.state in ("queued", "running")
                ):
                    return job
        return None

    def submit(self, kind, target, fn):
        existing = self.active(kind, target)
        if existing:
            return existing
        job = Job(kind, target)
        with self.lock:
            self.jobs[job.id] = job
            # Keep the registry small; the newest 50 are plenty for a UI.
            for old in sorted(self.jobs.values(), key=lambda j: j.started)[:-50]:
                self.jobs.pop(old.id, None)

        def run():
            job.state = "running"
            try:
                job.result = fn(job)
                job.state = "done"
                job.progress = 1.0
            except Exception as error:  # noqa: BLE001
                job.state = "failed"
                job.error = f"{type(error).__name__}: {error}"
                job.log(traceback.format_exc())
            job.finished = time.time()

        threading.Thread(target=run, daemon=True).start()
        return job

    def list(self):
        with self.lock:
            return [
                j.as_dict()
                for j in sorted(self.jobs.values(), key=lambda j: -j.started)
            ]

    def get(self, job_id):
        return self.jobs.get(job_id)
