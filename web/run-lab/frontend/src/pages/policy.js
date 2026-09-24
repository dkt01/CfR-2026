// How the RL driver behaved: outcome, where it lost margin, and whether its
// actions were healthy (saturation, residual vs prior, input freshness).

import { h, card, tile, fmt, statusBadge, clock } from "../ui.js";
import { timeChart, xyChart, lapBands } from "../charts.js";
import { seek } from "../playhead.js";

function lapProfiles(series, laps, key, reduce, bin = 0.5, length = 110.16) {
    const c = series.cols;
    const nb = Math.ceil(length / bin);
    const xs = Array.from({ length: nb }, (_, i) => +(i * bin + bin / 2).toFixed(2));
    const out = laps.map(() => Array(nb).fill(null));
    if (!c[key] || !c.station) return { xs, sets: out };
    laps.forEach((lap, k) => {
        const acc = Array.from({ length: nb }, () => []);
        for (let i = 0; i < series.n; i++) {
            const t = c.t[i];
            if (t < lap.t0 || t >= lap.t1) continue;
            const s = c.station[i];
            const v = c[key][i];
            if (Number.isNaN(s) || Number.isNaN(v)) continue;
            acc[Math.min(nb - 1, Math.floor(s / bin))].push(v);
        }
        acc.forEach((vals, b) => {
            if (vals.length) out[k][b] = reduce(vals);
        });
    });
    return { xs, sets: out };
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const min = (a) => Math.min(...a);

export default async function policyPage(root, { data, summary, series }) {
    const course = await data.course();
    const p = summary.policy || {};
    const laps = (summary.laps || {}).laps || [];
    const charts = [];
    const clr = p.clearance || {};
    const act = p.actions || {};

    const tiles = h(
        "div",
        { class: "tiles" },
        tile("Laps", `${p.laps_completed ?? 0} / ${p.laps_target ?? "?"}`, "", summary.laps?.finished ? "good" : "bad"),
        tile("Race time", fmt(p.race_time_s, 2), "s"),
        tile("Min clearance", fmt(clr.min, 3), "m", clr.min < 0 ? "bad" : clr.min < 0.12 ? "warn" : "good", clr.at_station !== undefined ? `station ${fmt(clr.at_station, 1)} m` : null),
        tile("In graze band", fmt(clr.time_in_graze_s, 1), "s", clr.time_in_graze_s > 0 ? "warn" : null),
        tile("Mean |CTE|", fmt(p.cte?.mean_abs, 3), "m", null, `max ${fmt(p.cte?.max_abs, 3)}`),
        tile("Heading RMS", fmt(p.heading_rms_deg, 1), "°"),
        tile("Residual RMS", fmt(act.residual_rms, 3), "", null, act.residual_share !== undefined ? `${fmt(act.residual_share * 100, 0)}% of steering` : null),
        tile("Action saturation", act.steer_saturated_pct !== undefined ? `${fmt(act.steer_saturated_pct, 1)} / ${fmt(act.throttle_saturated_pct, 1)}` : null, "% S/T"),
        tile("Speed input from tach", fmt(p.speed_from_tach_pct, 0), "%", p.speed_from_tach_pct !== undefined && p.speed_from_tach_pct < 95 ? "warn" : null, "else pose-differenced"),
        tile("Pose age p95", p.pose_age_p95_s !== undefined ? fmt(p.pose_age_p95_s * 1000, 0) : null, "ms"),
    );

    // Profiles by station, one line per lap.
    const lapColors = laps.map((_, k) => `--series-${(k % 8) + 1}`);
    const profile = (key, reduce, extra = [], opts = {}) => {
        const body = h("div");
        if (!laps.length || !series.cols[key]) {
            body.append(h("div", { class: "muted" }, "Needs at least one complete lap."));
            return body;
        }
        const { xs, sets } = lapProfiles(series, laps, key, reduce, 0.5, course.length);
        const lines = sets.map((y, k) => ({ label: `lap ${laps[k].lap}`, color: lapColors[k], x: xs, y }));
        for (const e of extra) lines.push({ ...e, x: xs });
        requestAnimationFrame(() => charts.push(xyChart(body, lines, { xlabel: "station (m)", ylabel: opts.ylabel, height: 230 })));
        return body;
    };
    // The cap and floor along the station axis, from the course.
    const capAt = (xs) => {
        const cl = course.centerline;
        return xs.map((s) => {
            let j = 0;
            while (j < cl.s.length - 1 && cl.s[j + 1] < s) j++;
            return cl.v_cap[j];
        });
    };
    const xsBins = Array.from({ length: Math.ceil(course.length / 0.5) }, (_, i) => +(i * 0.5 + 0.25).toFixed(2));
    const floorProfile = series.cols.v_floor ? lapProfiles(series, laps.slice(0, 1), "v_floor", mean, 0.5, course.length).sets[0] : null;

    // Section x lap matrix of minimum clearance.
    const sections = summary.sections || [];
    const matrix = h("table", { class: "data" });
    matrix.append(h("thead", {}, h("tr", {}, h("th", {}, "Section"), laps.map((l) => h("th", { class: "num" }, `Lap ${l.lap}`)), h("th", { class: "num" }, "Worst"))));
    const tb = h("tbody");
    const cell = (v, onClick) => {
        const tone = v === null || v === undefined ? "" : v < 0 ? "bad" : v < 0.12 ? "warn" : "";
        const td = h("td", { class: "num", style: { cursor: onClick ? "pointer" : "default" } }, tone ? statusBadge(tone === "bad" ? "bad" : "warn", fmt(v, 3)) : fmt(v, 3));
        if (onClick) td.addEventListener("click", onClick);
        return td;
    };
    for (const s of sections) {
        tb.append(
            h(
                "tr",
                {},
                h("td", {}, h("b", {}, s.name), h("span", { class: "muted small" }, `  ${fmt(((s.s0 % course.length) + course.length) % course.length, 0)}–${fmt(((s.s1 % course.length) + course.length) % course.length, 0)} m`)),
                laps.map((l) => {
                    const row = s.per_lap.find((x) => x.lap === l.lap);
                    return cell(row?.min_clearance, row ? () => seek(l.t0 + 0.01, "matrix") : null);
                }),
                cell(s.all.min_clearance),
            ),
        );
    }
    matrix.append(tb);

    root.append(
        h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "RL policy"), h("p", {}, "What the driver did with the car it was given.  Speed, clearance and line are overlaid lap by lap against station, so a lap that lifted early or ran wide shows up as the line that leaves the others."))),
        tiles,
        h(
            "div",
            { class: "grid cols-2", style: { marginTop: "16px" } },
            card("Speed profile by station", "per lap, with the rule cap", profile("speed", mean, [{ label: "cap", color: "--text-muted", y: capAt(xsBins) }, ...(floorProfile ? [{ label: "floor", color: "--series-4", y: floorProfile }] : [])], { ylabel: "m/s" })),
            card("Clearance by station", "per lap, minimum in each 0.5 m", profile("clearance", min, [{ label: "graze band", color: "--warning", y: xsBins.map(() => 0.12) }], { ylabel: "m" })),
            card("Cross-track error by station", "per lap, + is left", profile("cte", mean, [], { ylabel: "m" })),
            card("Steering by station", "per lap, command on the wire", profile("cmd_steer", mean, [], { ylabel: "normalised" })),
        ),
        h("div", { class: "grid cols-2", style: { marginTop: "16px" } }, card("Minimum clearance: section × lap", "click a cell to jump to that lap", h("div", { class: "table-wrap" }, matrix)), card("Actions over time", "raw network output", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "act_steer", label: "steer", color: "--series-1" }, { key: "act_throttle", label: "throttle", color: "--series-2" }], { range: [-1.05, 1.05], bands: lapBands(summary), height: 230 })); return b; })())),
        h("div", { style: { marginTop: "16px" } }, card("Prior vs residual", "the policy steers as a residual on a stabilising centerline prior", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "steer_ff", label: "prior", color: "--series-2" }, { key: "residual", label: "residual", color: "--series-3" }, { key: "cmd_steer", label: "total", color: "--series-1", width: 2 }], { range: [-1.05, 1.05], bands: lapBands(summary), height: 220 })); return b; })())),
    );
    void clock;
    return () => charts.forEach((c) => c && c.destroy());
}
