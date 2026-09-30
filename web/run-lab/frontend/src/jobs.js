// Polls /api/jobs while anything is running, and tells subscribers which
// jobs changed state.  Stops polling when everything is idle.

import { api } from "./api.js";

const listeners = new Set();
let known = null; // null until the first poll: jobs finished before this page loaded are not news
let timer = null;

export const jobs = {
    list: [],
    active(kind, target) {
        return this.list.find((j) => j.kind === kind && j.target === target && (j.state === "running" || j.state === "queued"));
    },
    onChange(fn) {
        listeners.add(fn);
        return () => listeners.delete(fn);
    },
    async poll() {
        clearTimeout(timer);
        try {
            this.list = await api.get("/api/jobs");
        } catch {
            this.list = [];
        }
        const changed = [];
        const first = known === null;
        for (const j of this.list) {
            if (first && (j.state === "done" || j.state === "failed")) continue;
            const prev = known?.get(j.id);
            // Progress/state alone misses a job that is only streaming log
            // lines (calibration.launch()'s ros2 launch output) without
            // moving progress until it finishes -- the log would sit frozen
            // the whole run. Lines are capped at 20 by the server, so once
            // past that the array LENGTH stops changing even as content
            // scrolls; compare the joined text instead.
            if (!prev || prev.state !== j.state || prev.progress !== j.progress || (prev.lines || []).join("\n") !== (j.lines || []).join("\n")) changed.push(j);
        }
        known = new Map(this.list.map((j) => [j.id, j]));
        if (changed.length) for (const fn of listeners) fn(changed);
        if (this.list.some((j) => j.state === "running" || j.state === "queued")) {
            timer = setTimeout(() => this.poll(), 800);
        }
    },
};
