import "uplot/dist/uPlot.min.css";
import "./style.css";
import { api, runData, dropRun } from "./api.js";
import { h, icon, toast, statusBadge, when, empty } from "./ui.js";
import { mountTimeline, setRange, setPlaying } from "./playhead.js";
import { jobs } from "./jobs.js";

import carPage from "./pages/car.js";
import runsPage from "./pages/runs.js";
import overviewPage from "./pages/overview.js";
import trackPage from "./pages/track.js";
import replayPage from "./pages/replay.js";
import vehiclePage from "./pages/vehicle.js";
import policyPage from "./pages/policy.js";
import perceptionPage from "./pages/perception.js";
import systemPage from "./pages/system.js";
import filesPage from "./pages/files.js";

// The run categories, in the order a run is usually read: verdict first,
// then where on the track, then why (vehicle / policy / localisation).
// Raw visualisation -- 3D, clouds, camera, every channel, logs -- is Rerun's
// job, on the Replay page; these pages are the analysis Rerun cannot do.
const RUN_TABS = [
    ["overview", "Overview", "overview", overviewPage],
    ["replay", "Replay (Rerun)", "replay", replayPage],
    ["track", "Track & sections", "track", trackPage],
    ["vehicle", "Vehicle model", "wheel", vehiclePage],
    ["policy", "RL policy", "brain", policyPage],
    ["perception", "ZED & localisation", "cube", perceptionPage],
    ["system", "System health", "cpu", systemPage],
    ["files", "Files", "files", filesPage],
];

const state = { health: null, runs: [], route: null, cleanup: null };

// ------------------------------------------------------------------ theme

function applyTheme(mode) {
    if (mode === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", mode);
    try {
        localStorage.setItem("runlab-theme", mode);
    } catch {}
}
let theme = "auto";
try {
    theme = localStorage.getItem("runlab-theme") || "auto";
} catch {}
applyTheme(theme);

// ------------------------------------------------------------------ route

function parseRoute() {
    const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
    if (parts[0] === "run" && parts[1]) return { page: "run", run: parts[1], tab: parts[2] || "overview" };
    if (parts[0] === "car") return { page: "car" };
    return { page: "runs" };
}

export function go(hash) {
    location.hash = hash;
}

window.addEventListener("hashchange", render);

// ---------------------------------------------------------------- sidebar

function renderSidebar() {
    const r = state.route || parseRoute();
    const sb = document.getElementById("sidebar");
    const item = (href, label, ic, active, count) =>
        h("a", { class: `nav-item ${active ? "active" : ""}`, href }, icon(ic), label, count !== undefined ? h("span", { class: "count" }, count) : null);
    const themeBtn = h("button", { class: "theme-btn", title: "Theme" }, { auto: "Auto", light: "Light", dark: "Dark" }[theme]);
    themeBtn.addEventListener("click", () => {
        theme = { auto: "light", light: "dark", dark: "auto" }[theme];
        applyTheme(theme);
        render();
    });
    sb.replaceChildren(
        ...[
        h("div", { class: "brand" }, h("div", { class: "brand-mark" }, "CfR"), h("div", {}, h("div", { class: "brand-name" }, "CfR Log Playback"), h("div", { class: "brand-sub" }, "logs · analysis · replay"))),
        h("div", { class: "nav-section" }, "Data"),
        item("#/car", "Car (Jetson)", "usb", r.page === "car"),
        item("#/runs", "Runs", "runs", r.page === "runs", state.runs.length),
        r.page === "run"
            ? [
                  h("div", { class: "nav-section" }, "This run"),
                  h("div", { class: "nav-run" }, runLabel(r.run), h("small", {}, r.run)),
                  RUN_TABS.map(([key, label, ic]) => item(`#/run/${encodeURIComponent(r.run)}/${key}`, label, ic, r.tab === key)),
              ]
            : null,
        h("div", { class: "sidebar-foot" }, state.health ? h("span", {}, state.health.ros_available ? "ROS 2 ✓" : "no ROS") : null, themeBtn),
        ].flat(Infinity).filter(Boolean),
    );
}

function runLabel(name) {
    const run = state.runs.find((x) => x.name === name);
    return run ? run.label : name;
}

// ----------------------------------------------------------------- topbar

function renderTopbar(extra) {
    const tb = document.getElementById("topbar");
    tb.replaceChildren(...extra.flat(Infinity).filter((x) => x !== null && x !== undefined && x !== false));
}

function runTopbar(run) {
    const bits = [
        h("div", {}, h("h1", {}, run.label), h("div", { class: "crumbs" }, run.name, run.started_utc ? ` · ${when(run.started_utc)}` : "")),
        run.simulation === true ? h("span", { class: "badge info" }, "simulation") : run.processed ? h("span", { class: "badge" }, "car") : null,
        run.driver ? h("span", { class: "badge" }, `driver: ${run.driver}`) : null,
        run.speed_scale && run.speed_scale !== "1.0" ? h("span", { class: "badge warn" }, `speed ×${run.speed_scale}`) : null,
        run.stale ? statusBadge("warn", "analysis out of date") : null,
        h("span", { class: "spacer" }),
    ];
    const job = jobs.active("process", run.name);
    if (job) {
        bits.push(h("div", { style: { width: "220px" } }, h("div", { class: "small muted" }, job.message || "processing…"), h("div", { class: "progress" }, h("div", { style: { width: `${job.progress * 100}%` } }))));
    }
    const process = h("button", { class: `btn ${run.processed ? "" : "primary"}`, disabled: !!job || !run.has_bag }, icon("refresh"), run.processed ? "Re-process" : "Process");
    process.addEventListener("click", async () => {
        await api.post(`/api/runs/${run.name}/process`, {});
        jobs.poll();
        toast("Processing started");
    });
    bits.push(
        process,
        run.processed ? h("a", { class: "btn", href: `/api/runs/${run.name}/report.md` }, icon("download"), "Report") : null,
        h("a", { class: "btn", href: `/api/runs/${run.name}/archive`, title: "The whole run as a .tar (bag, logs, params, policy)" }, icon("download"), "Download run"),
    );
    return bits;
}

// ----------------------------------------------------------------- render

async function render() {
    state.route = parseRoute();
    if (state.cleanup) {
        try {
            state.cleanup();
        } catch {}
        state.cleanup = null;
    }
    setPlaying(false);
    const page = document.getElementById("page");
    const timeline = document.getElementById("timeline");
    renderSidebar();
    const r = state.route;
    page.scrollTop = 0;

    if (r.page === "car") {
        timeline.hidden = true;
        renderTopbar([h("h1", {}, "Car (Jetson)"), h("span", { class: "spacer" })]);
        state.cleanup = await carPage(page, { go, refreshRuns });
        return;
    }
    if (r.page === "runs") {
        timeline.hidden = true;
        renderTopbar([h("h1", {}, "Runs"), h("span", { class: "spacer" })]);
        state.cleanup = await runsPage(page, { go, refreshRuns, runs: () => state.runs });
        return;
    }

    // a run
    let run;
    try {
        run = await api.get(`/api/runs/${encodeURIComponent(r.run)}`);
    } catch (e) {
        page.replaceChildren(empty("Run not found", e.message));
        return;
    }
    renderTopbar(runTopbar(run));
    if (!run.processed) {
        timeline.hidden = true;
        const job = jobs.active("process", run.name);
        page.replaceChildren(
            empty(
                job ? "Processing…" : "Not processed yet",
                job
                    ? job.message
                    : run.has_bag
                      ? "Processing reads the bag once and builds everything the other tabs show: the timeline, verdicts, the track map, the 3D map and the camera frames.  A two-lap run takes under a minute."
                      : "This run has no rosbag, so there is nothing to analyse.  Its files are still available under Files.",
                !job && run.has_bag
                    ? (() => {
                          const b = h("button", { class: "btn primary" }, icon("refresh"), "Process now");
                          b.addEventListener("click", async () => {
                              await api.post(`/api/runs/${run.name}/process`, {});
                              jobs.poll();
                              render();
                          });
                          return b;
                      })()
                    : null,
            ),
        );
        if (!run.has_bag) {
            const holder = h("div", { style: { marginTop: "20px" } });
            page.append(holder);
            await filesPage(holder, { run: r.run, data: runData(r.run) });
        }
        return;
    }

    const data = runData(r.run);
    const tab = RUN_TABS.find(([k]) => k === r.tab) || RUN_TABS[0];
    try {
        const [summary, series] = await Promise.all([data.summary(), data.series()]);
        const t0 = series.cols.t[0];
        const t1 = series.cols.t[series.n - 1];
        setRange(r.run, t0, t1, summary.events || [], (summary.laps || {}).laps || [], summary.window);
        timeline.hidden = false;
        page.replaceChildren();
        state.cleanup = await tab[3](page, { run: r.run, info: run, data, summary, series, go });
    } catch (e) {
        console.error(e);
        page.replaceChildren(empty("Could not load this run", e.message));
    }
}

async function refreshRuns() {
    try {
        state.runs = await api.get("/api/runs");
    } catch {
        state.runs = [];
    }
    renderSidebar();
    return state.runs;
}

// Re-render a run page when its processing job finishes.
jobs.onChange((changed) => {
    for (const job of changed) {
        if (job.kind === "process" && job.state === "done") {
            dropRun(job.target);
            toast(`Processed ${job.target}`, "good");
            refreshRuns();
            if (state.route?.page === "run" && state.route.run === job.target) render();
        } else if (job.kind === "process" && job.state === "failed") {
            toast(`Processing ${job.target} failed: ${job.error}`, "bad");
            if (state.route?.page === "run" && state.route.run === job.target) render();
        }
    }
    if (state.route?.page === "run") {
        const run = state.runs.find((x) => x.name === state.route.run);
        if (run) api.get(`/api/runs/${encodeURIComponent(run.name)}`).then((fresh) => renderTopbar(runTopbar(fresh))).catch(() => {});
    }
});

// Nothing in this UI may fail silently: surface every uncaught error.
window.addEventListener("error", (e) => toast(`UI error: ${e.message}`, "bad"));
window.addEventListener("unhandledrejection", (e) => toast(`UI error: ${e.reason?.message || e.reason}`, "bad"));

async function boot() {
    mountTimeline(document.getElementById("timeline"));
    try {
        state.health = await api.get("/api/health");
    } catch {}
    await refreshRuns();
    jobs.poll();
    render();
}

boot();
