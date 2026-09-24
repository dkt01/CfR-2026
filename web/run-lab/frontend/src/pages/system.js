// The platform under the driver: battery, the Arduino link, modes, the
// Orin's load and every topic's rate against what it should have been.

import { h, card, tile, fmt, table, statusBadge, kv } from "../ui.js";
import { timeChart, xyChart, lapBands } from "../charts.js";

export default async function systemPage(root, { summary, series }) {
    const sys = summary.system || {};
    const tg = sys.tegrastats;
    const charts = [];
    const lc = summary.log_counts || {};

    const tiles = h(
        "div",
        { class: "tiles" },
        tile("Battery", sys.battery ? `${fmt(sys.battery.start_v, 2)} → ${fmt(sys.battery.end_v, 2)}` : null, "V", null, sys.battery ? `min ${fmt(sys.battery.min_v, 2)} V` : "not recorded"),
        tile("Arduino link", fmt(sys.link_ok_pct, 1), "% ok", sys.link_ok_pct !== undefined && sys.link_ok_pct < 99.5 ? "bad" : "good"),
        tile("E-stops", sys.estop_events ?? 0, "", sys.estop_events ? "warn" : null),
        tile("CPU", tg?.cpu_pct ? `${fmt(tg.cpu_pct.mean, 0)} / ${fmt(tg.cpu_pct.max, 0)}` : null, "% avg/max", tg?.cpu_pct?.max > 95 ? "warn" : null, tg ? null : "no tegrastats (not on an Orin)"),
        tile("GPU", tg?.gpu_pct ? `${fmt(tg.gpu_pct.mean, 0)} / ${fmt(tg.gpu_pct.max, 0)}` : null, "% avg/max"),
        tile("Hottest", fmt(tg?.temp_max_c, 1), "°C", tg?.temp_max_c > 85 ? "bad" : null),
        tile("RAM peak", fmt(tg?.ram_pct_max, 0), "%"),
        tile("Log warnings / errors", `${lc.warn ?? 0} / ${(lc.error ?? 0) + (lc.fatal ?? 0)}`, "", (lc.error ?? 0) + (lc.fatal ?? 0) ? "bad" : lc.warn ? "warn" : "good"),
    );

    const health = summary.topic_health || [];
    const topicTable = table(
        [
            { label: "Topic", get: (r) => h("span", { class: "mono" }, r.topic) },
            { label: "Type", get: (r) => h("span", { class: "small muted" }, r.type) },
            { label: "Messages", num: true, get: (r) => r.count.toLocaleString() },
            { label: "Rate", num: true, get: (r) => `${fmt(r.rate_hz, 1)} Hz` },
            { label: "Expected", num: true, get: (r) => (r.expected_hz ? `≥ ${Math.round(r.expected_hz * 0.6)} Hz` : "") },
            { label: "", get: (r) => (r.ok === undefined ? "" : r.missing ? statusBadge("bad", "missing") : r.ok ? statusBadge("good", "ok") : statusBadge("warn", "slow")) },
        ],
        [...health].sort((a, b) => (a.ok === false ? -1 : 0) - (b.ok === false ? -1 : 0) || b.count - a.count),
    );

    const modes = sys.mode_time_s ? kv(Object.entries(sys.mode_time_s).map(([k, v]) => [k, `${fmt(v, 1)} s`])) : h("div", { class: "muted" }, "No Arduino status in this bag.");

    const tgBody = h("div");
    root.append(
        h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "System health"), h("p", {}, "The platform the driver ran on.  A slow topic, a link drop or a pack sagging under load explains many behaviours that look like policy faults."))),
        tiles,
        h(
            "div",
            { class: "grid cols-2", style: { marginTop: "16px" } },
            card("Battery", "approximate pack voltage", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "battery", label: "pack", color: "--series-1" }], { unit: "V", bands: lapBands(summary), height: 180 })); return b; })()),
            card("Arduino link & mode", null, h("div", { class: "stack", style: { gap: "10px" } }, (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "link_ok", label: "link ok", color: "--series-3" }, { key: "estop", label: "e-stop", color: "--series-8" }], { range: [-0.1, 1.1], height: 120, digits: 0 })); return b; })(), modes)),
        ),
        h("div", { style: { marginTop: "16px" } }, card("Orin load", tg ? `tegrastats, ${tg.samples} samples at 1 Hz` : "", tgBody)),
        h("div", { style: { marginTop: "16px" } }, card("Topics", `${health.length} in the bag`, topicTable)),
    );
    if (tg?.series?.cpu?.length) {
        const xs = tg.series.cpu.map((_, i) => i);
        charts.push(
            xyChart(
                tgBody,
                [
                    { label: "CPU %", color: "--series-1", x: xs, y: tg.series.cpu },
                    { label: "GPU %", color: "--series-2", x: xs, y: tg.series.gpu.slice(0, xs.length) },
                ],
                { xlabel: "seconds since recording started", ylabel: "%", height: 200 },
            ),
        );
    } else tgBody.append(h("div", { class: "muted" }, "tegrastats is only recorded on the Orin."));
    return () => charts.forEach((c) => c && c.destroy());
}
