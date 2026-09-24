// The ZED: how good the pose was -- the driver's only input.  The clouds and
// the map they build are viewed in Rerun (Replay page); this page judges the
// localisation.

import { h, card, tile, fmt, kv, statusBadge, icon } from "../ui.js";
import { timeChart, lapBands } from "../charts.js";

export default async function perceptionPage(root, { run, summary, series }) {
    const loc = summary.localization || {};
    const per = summary.perception || {};
    const charts = [];

    const tiles = h(
        "div",
        { class: "tiles" },
        tile("Pose rate", fmt(loc.rate_hz, 1), "Hz", loc.rate_hz < 10 ? "warn" : "good"),
        tile("Pose jumps", loc.jumps ?? "—", "", loc.jumps ? "bad" : "good", loc.jumps ? `largest ${fmt(loc.largest_jump_m, 2)} m` : "none"),
        tile("Longest gap", fmt(loc.max_gap_s, 3), "s", loc.max_gap_s > 0.3 ? "warn" : null),
        tile("Gaps > 100 ms", loc.gaps_over_100ms ?? "—", ""),
        tile("Odom vs pose", fmt(loc.odom_vs_pose_end_m, 2), "m", null, "at the end; odom is never loop-closed"),
        tile("Driver vs analyser", fmt(loc.driver_vs_analyser_m, 3), "m", null, "max disagreement on position"),
        tile("Cloud frames", per.clouds ?? 0, ""),
        tile("Map points", per.map_points ? per.map_points.toLocaleString() : "—", ""),
    );

    const notes = [];
    if (per.cloud_tf_note) notes.push(h("div", { class: "note warn" }, per.cloud_tf_note));
    if (!per.clouds) notes.push(h("div", { class: "note" }, "No point cloud in this bag.  On the car the recorder lowers the ZED cloud to 1 Hz for the run; if the ZED would not take the parameter it leaves the cloud out entirely (see recorder.log)."));
    if (!per.fused_points) notes.push(h("div", { class: "note" }, "No ZED spatial map in this bag.  Record with record_args:=\"--map\" to enable the ZED's own mapping for the run (costs GPU on the Orin)."));

    const status = loc.tracking_status_pct
        ? h(
              "div",
              {},
              Object.entries(loc.tracking_status_pct).map(([field, dist]) =>
                  h("div", { style: { marginBottom: "6px" } }, h("span", { class: "muted small" }, field), " ", Object.entries(dist).map(([v, pct]) => h("span", { class: "badge", style: { marginLeft: "4px" } }, `${v}: ${pct}%`))),
              ),
          )
        : h("div", { class: "muted small" }, "No /pose/status in this bag.");

    root.append(
        h(
            "div",
            { class: "page-head" },
            h("div", {}, h("h2", {}, "ZED & localisation"), h("p", {}, "The driver has no camera input: it drives the map against the ZED's pose.  If the pose is wrong, every clearance and every decision downstream is confidently wrong.  Check this page before trusting a clearance.")),
            h("span", { class: "spacer" }),
            h("a", { class: "btn primary", href: `#/run/${encodeURIComponent(run)}/replay` }, icon("cube"), "Clouds & map in Rerun"),
        ),
        tiles,
        notes.length ? h("div", { class: "stack", style: { marginTop: "12px", gap: "8px" } }, notes) : null,
        h(
            "div",
            { class: "grid cols-2", style: { marginTop: "16px" } },
            card(
                "Pose health",
                null,
                h(
                    "div",
                    { class: "stack", style: { gap: "10px" } },
                    kv([
                        ["Track frame", summary.frame.method],
                        ["Fit RMS", summary.frame.fit_rms_m !== null ? `${fmt(summary.frame.fit_rms_m, 3)} m` : "—"],
                        ["Anchors", summary.frame.anchors ?? 1],
                        ["Map → track rotation", summary.frame.theta_deg !== null ? `${fmt(summary.frame.theta_deg, 2)}°` : "—"],
                        ["Lap counter loop closures", loc.lap_counter_loop_closures ?? "—"],
                        ["Rejected line crossings", loc.lap_counter_rejected_crossings ?? "—"],
                    ]),
                    status,
                    loc.jumps ? statusBadge("bad", "Jumps are listed as events on the timeline") : null,
                ),
            ),
            card("Odometry vs pose", "distance between the ZED's odom and its loop-closed pose", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "odom_divergence", label: "odom − pose", color: "--series-1" }], { unit: "m", bands: lapBands(summary), height: 200 })); return b; })()),
            card("Pose age at the driver", "how stale the pose was when the policy acted on it", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "pose_age", label: "pose age", color: "--series-1" }], { unit: "s", bands: lapBands(summary), height: 200, digits: 3 })); return b; })()),
            card("Position: analyser vs driver", "the driver's belief against the analysis's track frame", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "tel_x", label: "driver x", color: "--series-2" }, { key: "x", label: "analyser x", color: "--series-1" }], { unit: "m", bands: lapBands(summary), height: 200 })); return b; })()),
        ),
    );
    return () => charts.forEach((c) => c && c.destroy());
}
