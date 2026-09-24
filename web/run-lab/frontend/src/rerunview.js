// The Rerun web viewer, embedded, on the CfR Log Playback's clock.
//
// The recording (analysis/recording.rrd) is written with recording_id = the
// run name and a "run" timeline in seconds -- the same t as the playhead --
// so syncing is a unit conversion: Rerun reports and takes nanoseconds.
// Both directions are wired: scrub either one and the other follows.

import { WebViewer } from "@rerun-io/web-viewer";
import { h } from "./ui.js";
import { playhead, seek, subscribe } from "./playhead.js";

const TIMELINE = "run";

function theme() {
    const forced = document.documentElement.getAttribute("data-theme");
    return forced === "light" || forced === "dark" ? forced : "system";
}

export class RerunView {
    constructor(container, { run, height = 720 }) {
        this.run = run;
        this.wrap = h("div", { class: "rerun-wrap", style: { height: `${height}px` } });
        this.status = h("div", { class: "rerun-status" }, "Loading the Rerun viewer…");
        this.wrap.append(this.status);
        container.append(this.wrap);
        this.viewer = new WebViewer();
        this.ready = false;
        this.echoUntil = 0;
        this.lastPush = 0;
        this.offs = [];
        this.start();
    }

    async start() {
        try {
            // Absolute: the viewer reads a bare "/api/..." as the host "api".
            const url = new URL(`/api/runs/${encodeURIComponent(this.run)}/recording.rrd`, location.href).href;
            await this.viewer.start(url, this.wrap, {
                hide_welcome_screen: true,
                width: "100%",
                height: "100%",
                theme: theme(),
            });
        } catch (e) {
            this.status.textContent = `The Rerun viewer did not start: ${e.message || e}`;
            this.status.classList.add("bad");
            return;
        }
        this.status.remove();
        // Rerun -> playhead.  Ignore the echo of our own set_current_time.
        this.offs.push(
            this.viewer.on("time_update", (event) => {
                if (performance.now() < this.echoUntil) return;
                // The callback gets the whole event: {recording_id, time (ns), ...}.
                if (event?.recording_id && event.recording_id !== this.run) return;
                const t = Number(typeof event === "object" ? event.time : event) / 1e9;
                if (Number.isFinite(t) && Math.abs(t - playhead.t) > 0.02) seek(t, "rerun");
            }),
        );
        this.offs.push(this.viewer.on("recording_open", () => this.push(playhead.t, true)));
        // Playhead -> Rerun.
        this.offs.push(
            subscribe((t, source) => {
                if (source === "rerun") return;
                this.push(t, source !== "play");
            }),
        );
        this.ready = true;
        this.push(playhead.t, true);
    }

    push(t, force) {
        if (!this.ready) return;
        const now = performance.now();
        // While our own clock plays, 20 updates a second is plenty.
        if (!force && now - this.lastPush < 50) return;
        this.lastPush = now;
        const id = this.viewer.get_active_recording_id() || this.run;
        this.echoUntil = now + 120;
        this.viewer.set_current_time(id, TIMELINE, Math.round(t * 1e9));
    }

    destroy() {
        for (const off of this.offs) {
            try {
                off();
            } catch {}
        }
        try {
            this.viewer.stop();
        } catch {}
    }
}
