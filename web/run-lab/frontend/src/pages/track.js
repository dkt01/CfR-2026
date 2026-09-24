// Where on the course: the path coloured by any metric, and the section
// table that says which part of the track cost the most.

import { h, card, fmt, table, segmented, statusBadge } from "../ui.js";
import { seek } from "../playhead.js";
import { TrackMap, METRICS } from "../trackmap.js";
import { indexAt } from "../api.js";

export default async function trackPage(root, { data, summary, series }) {
    const course = await data.course();
    const holder = h("div");
    let map;
    const pick = segmented(
        Object.entries(METRICS).map(([k, m]) => [k, m.label]),
        "speed",
        (k) => map.setMetric(k),
    );
    const follow = h("input", { type: "checkbox" });
    follow.addEventListener("change", () => {
        map.follow = follow.checked;
        if (follow.checked) map.view.scale = Math.max(map.view.scale, 60);
        map.draw();
    });
    const allTime = h("input", { type: "checkbox" });
    allTime.addEventListener("change", () => {
        map.onlyRun = !allTime.checked;
        map.draw();
    });

    const sections = summary.sections || [];
    const lapCount = ((summary.laps || {}).laps || []).length;
    let lapSel = "all";
    const secHolder = h("div");
    const risk = (r) => statusBadge(r === "bad" ? "bad" : r === "warn" ? "warn" : "good", r === "bad" ? "contact" : r === "warn" ? "graze" : "clear");
    function drawSections() {
        const rows = sections
            .map((s) => ({ ...s, row: lapSel === "all" ? s.all : (s.per_lap.find((p) => String(p.lap) === lapSel) || null) }))
            .filter((s) => s.row);
        secHolder.replaceChildren(
            table(
                [
                    { label: "Section", get: (s) => h("div", {}, h("b", {}, s.name), h("div", { class: "small muted" }, `${fmt(((s.s0 % course.length) + course.length) % course.length, 0)}–${fmt(((s.s1 % course.length) + course.length) % course.length, 0)} m`)) },
                    { label: "Risk", get: (s) => risk(s.risk) },
                    { label: "Time", num: true, get: (s) => `${fmt(s.row.time, 2)} s` },
                    { label: "Entry → exit", num: true, get: (s) => `${fmt(s.row.entry_speed, 2)} → ${fmt(s.row.exit_speed, 2)}` },
                    { label: "Min / max m/s", num: true, get: (s) => `${fmt(s.row.min_speed, 2)} / ${fmt(s.row.max_speed, 2)}` },
                    { label: "Cap", num: true, get: (s) => fmt(s.row.mean_cap, 2) },
                    { label: "Over cap", num: true, get: (s) => (s.row.max_over_cap > 0.05 ? h("span", { style: { color: "var(--critical-ink)" } }, `+${fmt(s.row.max_over_cap, 2)}`) : fmt(s.row.max_over_cap, 2)) },
                    { label: "Min clear", num: true, get: (s) => `${fmt(s.row.min_clearance, 3)} m` },
                    { label: "Max |CTE|", num: true, get: (s) => `${fmt(s.row.max_abs_cte, 3)} m` },
                    { label: "Steer RMS", num: true, get: (s) => fmt(s.row.steer_rms, 2) },
                ],
                rows,
                (s) => {
                    // Jump to where this section's minimum clearance happened.
                    const c = series.cols;
                    let best = null;
                    let bv = Infinity;
                    for (let i = 0; i < series.n; i++) {
                        const st = c.station?.[i];
                        if (st === undefined || Number.isNaN(st)) continue;
                        const s0 = ((s.s0 % course.length) + course.length) % course.length;
                        const s1 = ((s.s1 % course.length) + course.length) % course.length;
                        const inside = s0 < s1 ? st >= s0 && st < s1 : st >= s0 || st < s1;
                        if (!inside || c.t[i] < summary.window.t0 || c.t[i] > summary.window.t1) continue;
                        if (lapSel !== "all") {
                            const lap = summary.laps.laps[Number(lapSel) - 1];
                            if (c.t[i] < lap.t0 || c.t[i] >= lap.t1) continue;
                        }
                        const v = c.clearance?.[i] ?? 0;
                        if (v < bv) {
                            bv = v;
                            best = i;
                        }
                    }
                    if (best !== null) seek(c.t[best], "section");
                },
            ),
        );
    }
    const lapPick = segmented([["all", "All laps"], ...Array.from({ length: lapCount }, (_, i) => [String(i + 1), `Lap ${i + 1}`])], "all", (k) => {
        lapSel = k;
        drawSections();
    });

    root.append(
        h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "Track & sections"), h("p", {}, "The driven path on the surveyed course.  Colour it by any channel; hover for values, click to move the playhead, drag to pan, scroll to zoom.  The corridor is tinted by cap zone: red hairpin, amber taper, green straight."))),
        h(
            "section",
            { class: "card" },
            h("div", { class: "card-head" }, pick, h("span", { class: "spacer" }), h("label", { class: "check" }, follow, "Follow car"), h("label", { class: "check" }, allTime, "Include before/after the run")),
            h("div", { class: "card-body flush", style: { paddingTop: "10px" } }, holder),
        ),
        h("div", { style: { marginTop: "16px" } }, card("Sections", "ranked by the course, worst clearance flagged · click a row to jump to its tightest moment", h("div", { class: "stack", style: { gap: "10px" } }, lapPick, secHolder))),
    );
    map = new TrackMap(holder, { course, series, summary, metric: "speed", height: 600 });
    drawSections();
    void indexAt;
    return () => map.destroy();
}
