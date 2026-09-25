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
    // view: which layout the server's blueprint route builds -- "replay" (the
    // whole run) or "pose2d" (the ZED page's top-down pose).  height: a fixed
    // height in px; without it the viewer fills the page below its top edge.
    // stamp: when the run was last analysed.  It goes on every URL, so a
    // re-processed run is a new URL and no browser cache can hand back the
    // previous recording.
    constructor(container, { run, view = "replay", height = null, stamp = "" }) {
        this.run = run;
        this.view = view;
        this.stamp = stamp;
        this.wrap = h("div", { class: "rerun-wrap" });
        this.status = h("div", { class: "rerun-status" }, "Loading the Rerun viewer…");
        this.wrap.append(this.status);
        container.append(this.wrap);
        this.viewer = new WebViewer();
        this.ready = false;
        this.echoUntil = 0;
        this.lastPush = 0;
        this.offs = [];
        // Fill the scrolling page area (above the shared timeline) below the
        // viewer's top edge.
        this.fit = () => {
            if (height) {
                this.wrap.style.height = `${height}px`;
                return;
            }
            const page = document.getElementById("page");
            if (!page) return;
            const top = this.wrap.getBoundingClientRect().top - page.getBoundingClientRect().top + page.scrollTop;
            this.wrap.style.height = `${Math.max(560, page.clientHeight - top - 16)}px`;
        };
        this.fit();
        window.addEventListener("resize", this.fit);
        this.offs.push(() => window.removeEventListener("resize", this.fit));
        this.start();
    }

    async start() {
        try {
            // Absolute: the viewer reads a bare "/api/..." as the host "api".
            const url = new URL(`/api/runs/${encodeURIComponent(this.run)}/recording.rrd?v=${encodeURIComponent(this.stamp)}`, location.href).href;
            // The layout goes in with the recording, every time: blueprints
            // apply per application, so without it a page could open on the
            // layout another page last used.
            await this.viewer.start([url, this.blueprintUrl(false)], this.wrap, {
                hide_welcome_screen: true,
                allow_fullscreen: true,
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

    // Swap the layout for the one that opens on the "Follow car" view (or
    // back to the whole course).  Opening a blueprint applies it to the
    // recording with the same application id; the time cursor stays put,
    // but push it again in case the layout's time panel reset it.
    follow(on) {
        if (!this.ready) return;
        this.viewer.open(this.blueprintUrl(on));
        setTimeout(() => this.push(playhead.t, true), 400);
    }

    blueprintUrl(follow) {
        const q = `view=${encodeURIComponent(this.view)}&follow=${follow ? 1 : 0}&v=${encodeURIComponent(this.stamp)}`;
        return new URL(`/api/runs/${encodeURIComponent(this.run)}/blueprint.rbl?${q}`, location.href).href;
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
