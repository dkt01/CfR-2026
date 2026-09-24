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
            if (!prev || prev.state !== j.state || prev.progress !== j.progress) changed.push(j);
        }
        known = new Map(this.list.map((j) => [j.id, j]));
        if (changed.length) for (const fn of listeners) fn(changed);
        if (this.list.some((j) => j.state === "running" || j.state === "queued")) {
            timer = setTimeout(() => this.poll(), 800);
        }
    },
};
