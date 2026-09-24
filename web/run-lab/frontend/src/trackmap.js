// 2D course map on a canvas: bales, cap zones, the driven path coloured by a
// chosen metric, the car at the playhead.  Pan with drag, zoom with the
// wheel, hover for values, click to move the playhead there.

import { h, cssVar, fmt, clock } from "./ui.js";
import { indexAt } from "./api.js";
import { playhead, seek, subscribe } from "./playhead.js";

// Sequential blue, light -> dark (reference palette steps 100..700).
const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

function isDark() {
    return getComputedStyle(document.documentElement).colorScheme.includes("dark");
}

function lerpHex(a, b, f) {
    const pa = parseInt(a.slice(1), 16);
    const pb = parseInt(b.slice(1), 16);
    const ch = (s) => [(s >> 16) & 255, (s >> 8) & 255, s & 255];
    const [r1, g1, b1] = ch(pa);
    const [r2, g2, b2] = ch(pb);
    return `rgb(${Math.round(r1 + (r2 - r1) * f)},${Math.round(g1 + (g2 - g1) * f)},${Math.round(b1 + (b2 - b1) * f)})`;
}

function sequential(f) {
    // Keep the near-zero end visible against the surface: light mode starts
    // at step 250, dark mode runs 600 -> 150 (toward the light end = more).
    f = Math.max(0, Math.min(1, f));
    const steps = isDark() ? BLUE.slice(1, 11).reverse() : BLUE.slice(3);
    const x = f * (steps.length - 1);
    const i = Math.min(steps.length - 2, Math.floor(x));
    return lerpHex(steps[i], steps[i + 1], x - i);
}

function diverging(f) {
    // f in [-1, 1]: blue <- gray -> red
    f = Math.max(-1, Math.min(1, f));
    const mid = cssVar("--div-mid");
    const toHex = (c) => (c.startsWith("#") ? c : "#888888");
    return f < 0 ? lerpHex(toHex(mid), toHex(cssVar("--div-neg")), -f) : lerpHex(toHex(mid), toHex(cssVar("--div-pos")), f);
}

export const METRICS = {
    speed: {
        label: "Speed",
        unit: "m/s",
        value: (c, i) => Math.abs(c.speed ? c.speed[i] : c.speed_pose[i]),
        color: (v, ctx) => sequential(v / ctx.vmax),
        legend: (ctx) => ({ kind: "ramp", stops: [0, 0.5, 1].map((f) => sequential(f)), lo: "0", hi: `${ctx.vmax.toFixed(1)} m/s` }),
    },
    cap: {
        label: "Speed vs cap",
        unit: "m/s",
        value: (c, i) => (c.v_cap ? Math.abs(c.speed ? c.speed[i] : c.speed_pose[i]) - c.v_cap[i] : NaN),
        color: (v) => diverging(v / 1.5),
        legend: () => ({ kind: "ramp", stops: [-1, -0.5, 0, 0.5, 1].map(diverging), lo: "1.5 under", mid: "at cap", hi: "1.5 over" }),
    },
    clearance: {
        label: "Clearance to bales",
        unit: "m",
        value: (c, i) => (c.clearance ? c.clearance[i] : NaN),
        // Status, not magnitude: contact and the graze band are states.
        color: (v) => (v < 0 ? cssVar("--critical") : v < 0.12 ? cssVar("--warning") : v < 0.2 ? cssVar("--serious") : sequential(0.35)),
        legend: () => ({
            kind: "swatches",
            items: [
                [cssVar("--critical"), "contact < 0"],
                [cssVar("--warning"), "graze < 12 cm"],
                [cssVar("--serious"), "tight < 20 cm"],
                [sequential(0.35), "clear"],
            ],
        }),
    },
    cte: {
        label: "Cross-track error",
        unit: "m",
        value: (c, i) => (c.cte ? c.cte[i] : NaN),
        color: (v) => diverging(v / 0.25),
        legend: () => ({ kind: "ramp", stops: [-1, -0.5, 0, 0.5, 1].map(diverging), lo: "25 cm right", mid: "on line", hi: "25 cm left" }),
    },
    steer: {
        label: "Steering command",
        unit: "",
        value: (c, i) => (c.cmd_steer ? c.cmd_steer[i] : NaN),
        color: (v) => diverging(v),
        legend: () => ({ kind: "ramp", stops: [-1, -0.5, 0, 0.5, 1].map(diverging), lo: "full right", mid: "0", hi: "full left" }),
    },
    lap: {
        label: "Lap",
        unit: "",
        value: (c, i) => i,
        color: (v, ctx) => {
            const t = ctx.series.cols.t[v];
            const k = ctx.laps.findIndex((l) => t >= l.t0 && t < l.t1);
            return k < 0 ? cssVar("--text-muted") : cssVar(`--series-${(k % 8) + 1}`);
        },
        legend: (ctx) => ({ kind: "swatches", items: ctx.laps.map((l, k) => [cssVar(`--series-${(k % 8) + 1}`), `lap ${l.lap} · ${l.time.toFixed(2)} s`]) }),
    },
};

export class TrackMap {
    constructor(container, { course, series, summary, metric = "speed", onlyRun = true, height = 520, followCar = false }) {
        this.course = course;
        this.series = series;
        this.summary = summary;
        this.metric = metric;
        this.onlyRun = onlyRun;
        this.follow = followCar;
        this.wrap = h("div", { class: "map-wrap", style: { height: `${height}px` } });
        this.canvas = h("canvas");
        this.tip = h("div", { class: "map-tip", hidden: true });
        this.legend = h("div", { class: "map-legend" });
        const reset = h("button", { class: "btn sm", title: "Fit the course" }, "Fit");
        reset.addEventListener("click", () => {
            this.view = null;
            this.draw();
        });
        this.wrap.append(this.canvas, this.tip, this.legend, h("div", { class: "map-tools" }, reset));
        container.append(this.wrap);
        this.view = null; // {cx, cy, scale}
        this.vmax = Math.max(1, course.v_straight || 5.2);
        this.bindEvents();
        this.ro = new ResizeObserver(() => this.draw());
        this.ro.observe(this.wrap);
        this.unsub = subscribe(() => this.draw());
        this.draw();
    }

    destroy() {
        this.ro.disconnect();
        this.unsub();
    }

    setMetric(m) {
        this.metric = m;
        this.draw();
    }

    bounds() {
        const xs = this.course.centerline.x;
        const ys = this.course.centerline.y;
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (let i = 0; i < xs.length; i++) {
            x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]);
            y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]);
        }
        for (const b of this.course.bales) {
            x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x);
            y0 = Math.min(y0, b.y); y1 = Math.max(y1, b.y);
        }
        return { x0: x0 - 1, x1: x1 + 1, y0: y0 - 1, y1: y1 + 1 };
    }

    fit(w, ht) {
        const b = this.bounds();
        const scale = Math.min(w / (b.x1 - b.x0), ht / (b.y1 - b.y0));
        return { cx: (b.x0 + b.x1) / 2, cy: (b.y0 + b.y1) / 2, scale };
    }

    toScreen(x, y) {
        const v = this.view;
        return [this.w / 2 + (x - v.cx) * v.scale, this.h / 2 - (y - v.cy) * v.scale];
    }

    toWorld(px, py) {
        const v = this.view;
        return [v.cx + (px - this.w / 2) / v.scale, v.cy - (py - this.h / 2) / v.scale];
    }

    bindEvents() {
        let drag = null;
        this.canvas.addEventListener("pointerdown", (e) => {
            drag = { x: e.clientX, y: e.clientY, cx: this.view.cx, cy: this.view.cy, moved: false };
            this.canvas.setPointerCapture(e.pointerId);
        });
        this.canvas.addEventListener("pointermove", (e) => {
            if (drag) {
                const dx = e.clientX - drag.x;
                const dy = e.clientY - drag.y;
                if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
                this.view.cx = drag.cx - dx / this.view.scale;
                this.view.cy = drag.cy + dy / this.view.scale;
                this.follow = false;
                this.draw();
                return;
            }
            this.hover(e);
        });
        this.canvas.addEventListener("pointerup", (e) => {
            if (drag && !drag.moved) {
                const i = this.nearest(e);
                if (i !== null) seek(this.series.cols.t[i], "map");
            }
            drag = null;
        });
        this.canvas.addEventListener("pointerleave", () => (this.tip.hidden = true));
        this.canvas.addEventListener(
            "wheel",
            (e) => {
                e.preventDefault();
                const r = this.canvas.getBoundingClientRect();
                const [wx, wy] = this.toWorld(e.clientX - r.left, e.clientY - r.top);
                const k = Math.exp(-e.deltaY * 0.0015);
                this.view.scale = Math.max(5, Math.min(800, this.view.scale * k));
                const [nx, ny] = this.toWorld(e.clientX - r.left, e.clientY - r.top);
                this.view.cx += wx - nx;
                this.view.cy += wy - ny;
                this.draw();
            },
            { passive: false },
        );
    }

    inWindow(i) {
        if (!this.onlyRun || !this.summary.window) return true;
        const t = this.series.cols.t[i];
        return t >= this.summary.window.t0 - 0.5 && t <= this.summary.window.t1 + 0.5;
    }

    nearest(e) {
        const r = this.canvas.getBoundingClientRect();
        const px = e.clientX - r.left;
        const py = e.clientY - r.top;
        const { x, y } = this.series.cols;
        if (!x) return null;
        let best = null;
        let bd = 12 * 12;
        for (let i = 0; i < x.length; i++) {
            if (Number.isNaN(x[i]) || !this.inWindow(i)) continue;
            const [sx, sy] = this.toScreen(x[i], y[i]);
            const d = (sx - px) ** 2 + (sy - py) ** 2;
            if (d < bd) {
                bd = d;
                best = i;
            }
        }
        return best;
    }

    hover(e) {
        const i = this.nearest(e);
        if (i === null) {
            this.tip.hidden = true;
            return;
        }
        const c = this.series.cols;
        const r = this.canvas.getBoundingClientRect();
        const m = METRICS[this.metric];
        this.tip.hidden = false;
        this.tip.replaceChildren(
            h("div", {}, h("b", {}, clock(c.t[i])), c.station ? `  ·  station ${fmt(c.station[i], 1)} m` : ""),
            h("div", {}, `speed ${fmt(Math.abs(c.speed ? c.speed[i] : c.speed_pose[i]), 2)}`, c.v_cap ? ` / cap ${fmt(c.v_cap[i], 2)} m/s` : ""),
            c.clearance ? h("div", {}, `clearance ${fmt(c.clearance[i], 3)} m · CTE ${fmt(c.cte?.[i], 3)} m`) : null,
            c.cmd_steer ? h("div", {}, `steer ${fmt(c.cmd_steer[i], 2)}`) : null,
            h("div", { class: "muted small" }, "click to jump here"),
        );
        const px = e.clientX - r.left;
        const py = e.clientY - r.top;
        this.tip.style.left = `${Math.min(px + 14, this.w - 200)}px`;
        this.tip.style.top = `${Math.max(4, py - 70)}px`;
        void m;
    }

    draw() {
        const dpr = window.devicePixelRatio || 1;
        this.w = this.wrap.clientWidth;
        this.h = this.wrap.clientHeight;
        if (!this.w || !this.h) return;
        this.canvas.width = this.w * dpr;
        this.canvas.height = this.h * dpr;
        this.canvas.style.width = `${this.w}px`;
        this.canvas.style.height = `${this.h}px`;
        if (!this.view) this.view = this.fit(this.w, this.h);
        const c = this.series.cols;
        const iNow = indexAt(this.series, playhead.t);
        if (this.follow && c.x && !Number.isNaN(c.x[iNow])) {
            this.view.cx = c.x[iNow];
            this.view.cy = c.y[iNow];
        }
        const g = this.canvas.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.fillStyle = cssVar("--surface-1");
        g.fillRect(0, 0, this.w, this.h);
        const S = this.view.scale;

        // Corridor: the centerline, thick and faint, tinted by cap zone.
        const cl = this.course.centerline;
        const zoneColor = [cssVar("--critical"), cssVar("--warning"), cssVar("--good")];
        g.lineCap = "round";
        g.lineJoin = "round";
        g.lineWidth = Math.max(2, 0.9 * S);
        g.globalAlpha = 0.06;
        for (let i = 1; i < cl.x.length; i++) {
            g.strokeStyle = zoneColor[cl.zone[i]];
            g.beginPath();
            g.moveTo(...this.toScreen(cl.x[i - 1], cl.y[i - 1]));
            g.lineTo(...this.toScreen(cl.x[i], cl.y[i]));
            g.stroke();
        }
        g.globalAlpha = 1;
        g.strokeStyle = cssVar("--border");
        g.lineWidth = 1;
        g.setLineDash([]);
        g.beginPath();
        for (let i = 0; i < cl.x.length; i++) {
            const [sx, sy] = this.toScreen(cl.x[i], cl.y[i]);
            i ? g.lineTo(sx, sy) : g.moveTo(sx, sy);
        }
        g.closePath();
        g.stroke();

        // Bales.
        const [bl, bw] = this.course.bale_size;
        g.fillStyle = isDark() ? "#5b513a" : "#d9c99a";
        g.strokeStyle = isDark() ? "#76694a" : "#b9a66f";
        for (const b of this.course.bales) {
            const [sx, sy] = this.toScreen(b.x, b.y);
            g.save();
            g.translate(sx, sy);
            g.rotate(-b.yaw);
            g.fillRect((-bl / 2) * S, (-bw / 2) * S, bl * S, bw * S);
            g.lineWidth = 1;
            g.strokeRect((-bl / 2) * S, (-bw / 2) * S, bl * S, bw * S);
            g.restore();
        }

        // Start line.
        const st = this.course.start;
        const [sx, sy] = this.toScreen(st.x, st.y);
        g.save();
        g.translate(sx, sy);
        g.rotate(-st.yaw);
        g.fillStyle = cssVar("--text-primary");
        g.fillRect(-1.5, -0.46 * S, 3, 0.92 * S);
        g.restore();

        // Driven path.
        const metric = METRICS[this.metric];
        const ctx = { vmax: this.vmax, series: this.series, laps: (this.summary.laps || {}).laps || [] };
        if (c.x) {
            g.lineWidth = Math.max(2.2, Math.min(5, 0.07 * S));
            for (let i = 1; i < c.x.length; i++) {
                if (Number.isNaN(c.x[i]) || Number.isNaN(c.x[i - 1]) || !this.inWindow(i)) continue;
                const v = metric.value(c, i);
                g.strokeStyle = Number.isNaN(v) ? cssVar("--text-muted") : metric.color(v, ctx);
                g.beginPath();
                g.moveTo(...this.toScreen(c.x[i - 1], c.y[i - 1]));
                g.lineTo(...this.toScreen(c.x[i], c.y[i]));
                g.stroke();
            }
        }

        // Contacts and grazes, marked where they peaked.
        for (const ev of this.summary.events || []) {
            if (!["contact", "graze", "stall", "pose"].includes(ev.kind) || ev.t === null) continue;
            const i = indexAt(this.series, ev.t);
            if (!c.x || Number.isNaN(c.x[i])) continue;
            const [ex, ey] = this.toScreen(c.x[i], c.y[i]);
            g.beginPath();
            g.arc(ex, ey, ev.severity === "bad" ? 6 : 4.5, 0, Math.PI * 2);
            g.fillStyle = cssVar(ev.severity === "bad" ? "--critical" : "--warning");
            g.fill();
            g.lineWidth = 2;
            g.strokeStyle = cssVar("--surface-1");
            g.stroke();
        }

        // Car at the playhead: the footprint, pointing where it points.
        if (c.x && !Number.isNaN(c.x[iNow])) {
            const [cx, cy] = this.toScreen(c.x[iNow], c.y[iNow]);
            const [len, wid] = this.course.car || [0.58, 0.32];
            g.save();
            g.translate(cx, cy);
            g.rotate(-c.yaw[iNow]);
            const L = Math.max(14, len * S);
            const W = Math.max(8, wid * S);
            g.fillStyle = cssVar("--text-primary");
            g.strokeStyle = cssVar("--surface-1");
            g.lineWidth = 2;
            g.beginPath();
            g.roundRect(-L / 2, -W / 2, L, W, 3);
            g.fill();
            g.stroke();
            g.fillStyle = cssVar("--accent");
            g.beginPath();
            g.moveTo(L / 2, 0);
            g.lineTo(L / 2 - Math.min(10, L / 3), -W / 2 + 1);
            g.lineTo(L / 2 - Math.min(10, L / 3), W / 2 - 1);
            g.closePath();
            g.fill();
            g.restore();
        }
        this.drawLegend(metric, ctx);
    }

    drawLegend(metric, ctx) {
        const L = metric.legend(ctx);
        const items = [h("div", { style: { fontWeight: 600 } }, metric.label)];
        if (L.kind === "ramp") {
            items.push(h("div", { class: "ramp", style: { background: `linear-gradient(90deg, ${L.stops.join(",")})` } }));
            items.push(h("div", { class: "ends" }, h("span", {}, L.lo), L.mid ? h("span", {}, L.mid) : null, h("span", {}, L.hi)));
        } else {
            for (const [col, label] of L.items) {
                items.push(h("div", { class: "row", style: { gap: "6px", marginTop: "4px" } }, h("span", { style: { width: "14px", height: "4px", borderRadius: "2px", background: col, display: "inline-block" } }), label));
            }
        }
        items.push(h("div", { class: "row", style: { gap: "6px", marginTop: "6px", color: "var(--text-muted)" } }, "● event (contact / graze / stall)"));
        this.legend.replaceChildren(...items);
    }
}
