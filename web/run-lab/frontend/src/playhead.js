// The one piece of shared state: where in the run we are looking.
// Every view subscribes; the timeline dock, charts, map and event lists all
// write to it, so clicking anything moves everything.

import { h, icon, clock, cssVar } from "./ui.js";

const listeners = new Set();
export const playhead = {
    t: 0,
    t0: 0,
    t1: 1,
    playing: false,
    rate: 1,
    run: null,
    events: [],
    laps: [],
    window: null,
};

export function subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
}

export function seek(t, source) {
    playhead.t = Math.max(playhead.t0, Math.min(playhead.t1, t));
    for (const fn of listeners) fn(playhead.t, source);
}

export function setRange(run, t0, t1, events = [], laps = [], window = null) {
    const fresh = playhead.run !== run;
    playhead.run = run;
    playhead.t0 = t0;
    playhead.t1 = t1;
    playhead.events = events;
    playhead.laps = laps;
    playhead.window = window;
    if (fresh) {
        playhead.playing = false;
        seek(window ? window.t0 : t0, "init");
    }
    drawScrub();
}

let last = null;
function frame(ts) {
    if (!playhead.playing) {
        last = null;
        return;
    }
    if (last !== null) {
        const t = playhead.t + ((ts - last) / 1000) * playhead.rate;
        if (t >= playhead.t1) {
            seek(playhead.t1, "play");
            setPlaying(false);
            return;
        }
        seek(t, "play");
    }
    last = ts;
    requestAnimationFrame(frame);
}

export function setPlaying(on) {
    playhead.playing = on;
    renderPlayButton();
    if (on) requestAnimationFrame(frame);
}

// ------------------------------------------------------------ dock

let dock, scrubCanvas, clockEl, playBtn;

export function mountTimeline(root) {
    dock = root;
    playBtn = h("button", { class: "play-btn", title: "Play / pause (space)" });
    playBtn.addEventListener("click", () => setPlaying(!playhead.playing));
    clockEl = h("div", { class: "clock" });
    scrubCanvas = h("canvas");
    const scrub = h("div", { class: "scrub" }, scrubCanvas);
    const rate = h(
        "select",
        { title: "Playback rate" },
        [0.25, 0.5, 1, 2, 4, 8].map((r) => h("option", { value: r, selected: r === 1 }, `${r}×`)),
    );
    rate.addEventListener("change", () => (playhead.rate = Number(rate.value)));
    root.append(playBtn, clockEl, scrub, rate);
    renderPlayButton();

    let dragging = false;
    const toT = (e) => {
        const r = scrubCanvas.getBoundingClientRect();
        const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
        return playhead.t0 + f * (playhead.t1 - playhead.t0);
    };
    scrub.addEventListener("pointerdown", (e) => {
        dragging = true;
        scrub.setPointerCapture(e.pointerId);
        seek(toT(e), "scrub");
    });
    scrub.addEventListener("pointermove", (e) => dragging && seek(toT(e), "scrub"));
    scrub.addEventListener("pointerup", () => (dragging = false));
    new ResizeObserver(drawScrub).observe(scrub);
    subscribe(() => {
        drawScrub();
        clockEl.innerHTML = `${clock(playhead.t)} <small>/ ${clock(playhead.t1)}</small>`;
    });

    window.addEventListener("keydown", (e) => {
        if (dock.hidden || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
        if (e.code === "Space") {
            e.preventDefault();
            setPlaying(!playhead.playing);
        } else if (e.code === "ArrowRight") seek(playhead.t + (e.shiftKey ? 5 : 0.5), "key");
        else if (e.code === "ArrowLeft") seek(playhead.t - (e.shiftKey ? 5 : 0.5), "key");
    });
}

function renderPlayButton() {
    if (!playBtn) return;
    playBtn.replaceChildren(icon(playhead.playing ? "pause" : "play"));
}

const SEV = { bad: "--critical", warn: "--warning", good: "--good", info: "--text-muted" };

function drawScrub() {
    if (!scrubCanvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = scrubCanvas.clientWidth;
    const ht = scrubCanvas.clientHeight;
    if (!w) return;
    scrubCanvas.width = w * dpr;
    scrubCanvas.height = ht * dpr;
    const g = scrubCanvas.getContext("2d");
    g.scale(dpr, dpr);
    const span = playhead.t1 - playhead.t0 || 1;
    const X = (t) => ((t - playhead.t0) / span) * w;
    // track
    g.fillStyle = cssVar("--surface-3");
    g.fillRect(0, 14, w, 6);
    // run window
    if (playhead.window) {
        g.fillStyle = cssVar("--accent-soft");
        g.fillRect(X(playhead.window.t0), 10, X(playhead.window.t1) - X(playhead.window.t0), 14);
    }
    // laps: alternating bands above
    playhead.laps.forEach((lap, i) => {
        g.fillStyle = i % 2 ? cssVar("--series-3") : cssVar("--series-1");
        g.globalAlpha = 0.55;
        g.fillRect(X(lap.t0), 14, Math.max(1, X(lap.t1) - X(lap.t0) - 2), 6);
        g.globalAlpha = 1;
    });
    // events as ticks, coloured by status
    for (const ev of playhead.events) {
        if (ev.t === null || ev.severity === "info") continue;
        g.fillStyle = cssVar(SEV[ev.severity] || "--text-muted");
        g.fillRect(X(ev.t) - 1, ev.severity === "bad" ? 2 : 5, 2, ev.severity === "bad" ? 10 : 7);
    }
    // playhead
    const x = X(playhead.t);
    g.fillStyle = cssVar("--text-primary");
    g.fillRect(x - 1, 4, 2, 26);
    g.beginPath();
    g.arc(x, 17, 5, 0, Math.PI * 2);
    g.fill();
}
