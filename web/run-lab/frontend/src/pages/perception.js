// The ZED: how good the pose was -- the driver's only input.  The clouds and
// the map they build are viewed in Rerun (Replay page); this page judges the
// localisation.

import { h, card, tile, fmt, kv, statusBadge, icon } from "../ui.js";
import { timeChart, lapBands } from "../charts.js";
import { RerunView } from "../rerunview.js";

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
        tile("ZED map", per.fused_points ? per.fused_points.toLocaleString() : "—", per.fused_points ? "pts" : "", null, "the finished spatial map, saved at the end of the run"),
        tile("Cloud frames", per.clouds ?? 0, "", null, per.map_points ? `accumulated into ${per.map_points.toLocaleString()} pts` : "live cloud, off by default"),
    );

    const notes = [];
    if (per.cloud_tf_note) notes.push(h("div", { class: "note warn" }, per.cloud_tf_note));
    if (!per.fused_points) notes.push(h("div", { class: "note" }, "No ZED spatial map in this bag.  The recorder builds one by default and saves the finished map at the end of the run; it was either run with --no-map, or the map did not arrive in time (recorder.log says which)."));
    if (!per.clouds && !per.fused_points) notes.push(h("div", { class: "note" }, "No live point cloud either.  Add it with record_args:=\"--cloud-hz 1\"."));

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
        (() => {
            const holder = h("div");
            const hasPose = (series.cols.x_map || []).some((v) => Number.isFinite(v));
            // The pose layers live in recording.rrd, which only analyses from
            // version 6 on contain.
            const current = ((summary.meta || {}).version || 0) >= 6;
            const body = !hasPose
                ? h("div", { class: "muted small" }, "No /zed/zed_node/pose in this bag.")
                : !current
                  ? h("div", { class: "note warn" }, "This run was analysed before the 2D pose existed.  Re-process it (Runs list → Re-process) to plot it.")
                  : holder;
            const section = h(
                "section",
                { class: "card", style: { marginTop: "16px" } },
                h(
                    "div",
                    { class: "card-head", title: "The ZED's pose exactly as reported, in its own map frame and before any fit to the track -- seen from above, x right, y up.  A red segment is a step the car cannot make.  The arrow is the car at the playhead; scrub either timeline." },
                    h("h3", {}, "2D pose"),
                    h("span", { class: "sub" }, "ZED map frame · red jumps · arrow at the playhead"),
                ),
                h("div", { class: "card-body flush", style: { paddingTop: "10px" } }, body),
            );
            if (hasPose && current) charts.push(new RerunView(holder, { run, view: "pose2d", height: 460, stamp: (summary.meta || {}).processed_utc || "" }));
            return section;
        })(),
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
