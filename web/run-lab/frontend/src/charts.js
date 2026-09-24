// Time-series charts on uPlot.  One y-axis per chart, always: two measures
// of different units go in two charts, stacked and cursor-synced, never on a
// dual axis.  Every chart draws the playhead and seeks on click.

import uPlot from "uplot";
import { h, cssVar, clock } from "./ui.js";
import { playhead, seek, subscribe } from "./playhead.js";

const charts = new Set();
let unsub = null;

function ensureSubscription() {
    if (unsub) return;
    unsub = subscribe(() => {
        for (const u of charts) u.redraw(false, false);
    });
}

// Shade the run's laps and the graze band etc. behind the data.
function bandsPlugin(bands) {
    return {
        hooks: {
            drawClear: (u) => {
                const g = u.ctx;
                for (const b of bands || []) {
                    const x0 = u.valToPos(b.t0, "x", true);
                    const x1 = u.valToPos(b.t1, "x", true);
                    g.fillStyle = b.color;
                    g.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
                }
            },
        },
    };
}

function playheadPlugin() {
    return {
        hooks: {
            draw: (u) => {
                const x = u.valToPos(playhead.t, "x", true);
                if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) return;
                const g = u.ctx;
                g.save();
                g.strokeStyle = cssVar("--text-primary");
                g.lineWidth = 1.5 * devicePixelRatio;
                g.beginPath();
                g.moveTo(x, u.bbox.top);
                g.lineTo(x, u.bbox.top + u.bbox.height);
                g.stroke();
                g.restore();
            },
            ready: (u) => {
                u.over.addEventListener("click", () => {
                    if (u.select.width > 2) return;
                    const t = u.posToVal(u.cursor.left, "x");
                    if (Number.isFinite(t)) seek(t, "chart");
                });
                u.over.addEventListener("dblclick", () => u.setScale("x", { min: u.data[0][0], max: u.data[0][u.data[0].length - 1] }));
            },
        },
    };
}

// Horizontal reference lines (the graze band, zero, a limit).
function refLinesPlugin(lines) {
    return {
        hooks: {
            draw: (u) => {
                const g = u.ctx;
                for (const l of lines || []) {
                    const y = u.valToPos(l.y, "y", true);
                    if (y < u.bbox.top || y > u.bbox.top + u.bbox.height) continue;
                    g.save();
                    g.strokeStyle = l.color;
                    g.lineWidth = 1 * devicePixelRatio;
                    g.beginPath();
                    g.moveTo(u.bbox.left, y);
                    g.lineTo(u.bbox.left + u.bbox.width, y);
                    g.stroke();
                    if (l.label) {
                        g.fillStyle = l.color;
                        g.font = `${11 * devicePixelRatio}px Inter, sans-serif`;
                        g.fillText(l.label, u.bbox.left + 6 * devicePixelRatio, y - 4 * devicePixelRatio);
                    }
                    g.restore();
                }
            },
        },
    };
}

/**
 * lines: [{ key, label, color (css var name), width, dash }]
 * opts:  { height, unit, bands, refs, range: [min, max], digits }
 */
export function timeChart(container, series, lines, opts = {}) {
    ensureSubscription();
    const present = lines.filter((l) => series.cols[l.key]);
    if (!present.length) {
        container.append(h("div", { class: "muted small" }, "Not in this bag."));
        return null;
    }
    const el = h("div", { class: "chart" });
    container.append(el);
    // uPlot draws gaps for null, not NaN.
    const gaps = (arr) => Array.from(arr, (v) => (Number.isNaN(v) ? null : v));
    const data = [Array.from(series.cols.t), ...present.map((l) => gaps(l.transform ? series.cols[l.key].map(l.transform) : series.cols[l.key]))];
    const axisStroke = cssVar("--text-muted");
    const grid = { stroke: cssVar("--grid"), width: 1 };
    const digits = opts.digits ?? 2;
    const u = new uPlot(
        {
            width: el.clientWidth || 600,
            height: opts.height || 200,
            cursor: { sync: { key: "run", setSeries: false }, drag: { x: true, y: false, setScale: true }, points: { size: 7 } },
            legend: { live: true },
            scales: { x: { time: false }, y: opts.range ? { range: opts.range } : { auto: true } },
            axes: [
                { stroke: axisStroke, grid, ticks: { show: false }, values: (_, ticks) => ticks.map((t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`), size: 30 },
                { stroke: axisStroke, grid, ticks: { show: false }, size: 52, label: opts.unit || "", labelSize: 14, values: (_, ticks) => ticks.map((v) => v.toFixed(Math.abs(v) < 10 ? Math.min(digits, 2) : 0)) },
            ],
            series: [
                { label: "t", value: (_, v) => (v === null ? "—" : clock(v)) },
                ...present.map((l) => ({
                    label: l.label,
                    stroke: cssVar(l.color),
                    width: l.width || 1.6,
                    dash: l.dash,
                    points: { show: false },
                    spanGaps: false,
                    value: (_, v) => (v === null || Number.isNaN(v) ? "—" : `${v.toFixed(digits)}${opts.unit ? ` ${opts.unit}` : ""}`),
                })),
            ],
            plugins: [bandsPlugin(opts.bands), refLinesPlugin(opts.refs), playheadPlugin()],
        },
        data,
        el,
    );
    charts.add(u);
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height: opts.height || 200 }));
    ro.observe(el);
    return {
        u,
        destroy() {
            ro.disconnect();
            charts.delete(u);
            u.destroy();
        },
    };
}

// Scatter/line of y against an arbitrary x (not time), for model plots.
export function xyChart(container, sets, opts = {}) {
    const el = h("div", { class: "chart" });
    container.append(el);
    const axisStroke = cssVar("--text-muted");
    const grid = { stroke: cssVar("--grid"), width: 1 };
    // uPlot wants one shared x; merge the sets' x values.
    const xs = [...new Set(sets.flatMap((s) => s.x))].sort((a, b) => a - b);
    const data = [xs, ...sets.map((s) => xs.map((x) => { const i = s.x.indexOf(x); return i < 0 ? null : s.y[i]; }))];
    const u = new uPlot(
        {
            width: el.clientWidth || 400,
            height: opts.height || 240,
            legend: { live: true },
            cursor: { points: { size: 8 } },
            scales: { x: { time: false } },
            axes: [
                { stroke: axisStroke, grid, ticks: { show: false }, label: opts.xlabel || "", labelSize: 16, size: 40 },
                { stroke: axisStroke, grid, ticks: { show: false }, label: opts.ylabel || "", labelSize: 16, size: 52 },
            ],
            series: [
                { label: opts.xlabel || "x" },
                ...sets.map((s) => {
                    const def = {
                        label: s.label,
                        stroke: cssVar(s.color),
                        width: s.points ? 0 : 2,
                        spanGaps: true,
                        points: s.points ? { show: true, size: 8, fill: cssVar(s.color), stroke: cssVar("--surface-1"), width: 2 } : { show: false },
                    };
                    if (s.points) def.paths = () => null;
                    return def;
                }),
            ],
        },
        data,
        el,
    );
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height: opts.height || 240 }));
    ro.observe(el);
    return { u, destroy: () => { ro.disconnect(); u.destroy(); } };
}

export function lapBands(summary) {
    const laps = (summary.laps || {}).laps || [];
    return laps.map((l, i) => ({ t0: l.t0, t1: l.t1, color: i % 2 ? "rgba(128,128,128,0.06)" : "rgba(128,128,128,0)" }));
}
