// The car against the model the policy was trained on: speed envelope,
// speed loop, coast-down, steering authority and the steering-to-yaw lag.

import { h, card, tile, fmt, table, kv, statusBadge } from "../ui.js";
import { timeChart, xyChart, lapBands } from "../charts.js";

export default async function vehiclePage(root, { summary, series }) {
    const sp = summary.speed || {};
    const st = summary.steering || {};
    const lag = st.yaw_lag;
    const charts = [];

    const speedTiles = h(
        "div",
        { class: "tiles" },
        tile("Max speed", fmt(sp.max, 2), "m/s"),
        tile("Moving mean", fmt(sp.moving?.mean, 2), "m/s", null, `p95 ${fmt(sp.moving?.p95, 2)}`),
        tile("Min while racing", fmt(sp.min_while_racing, 2), "m/s"),
        tile("Max accel", fmt(sp.accel?.max_accel, 2), "m/s²"),
        tile("Max decel", fmt(sp.accel?.max_decel, 2), "m/s²", null, "no brakes: coast only"),
        tile("Time moving", fmt(sp.time_moving_s, 1), "s"),
        tile("Use of cap", sp.cap?.mean_use_of_cap !== undefined ? fmt(sp.cap.mean_use_of_cap * 100, 0) : null, "%"),
        tile("Over cap", fmt(sp.cap?.max_over, 2), "m/s", sp.cap?.max_over > 0.15 ? "bad" : "good", `${fmt(sp.cap?.time_over_s, 1)} s over`),
    );

    const tr = sp.tracking || {};
    const trackingKv = kv([
        ["Command → measured", `${fmt(tr.cmd_to_measured_rms, 2)} m/s RMS · ${fmt(tr.cmd_to_measured_lag_s, 2)} s lag`],
        ["Command → Arduino target", tr.cmd_to_target_rms !== undefined ? `${fmt(tr.cmd_to_target_rms, 2)} m/s RMS (slew limit ${tr.slew_limit} m/s², by design)` : undefined],
        ["Target → measured", tr.target_to_measured_rms !== undefined ? `${fmt(tr.target_to_measured_rms, 2)} m/s RMS · ${fmt(tr.target_to_measured_lag_s, 2)} s lag` : undefined],
        ["Plant dead time", tr.model_dead_time_s !== undefined ? `${tr.model_dead_time_s} s` : undefined],
        ["ZED / tachometer", sp.pose_vs_tach ? `${fmt(sp.pose_vs_tach.median_ratio, 3)} ×` : undefined],
        ["Coast-down fit", sp.coast ? sp.coast.fit : "not enough coasting"],
        ["Coast decel at 2 m/s", sp.coast ? `${fmt(sp.coast.decel_at_2ms, 3)} measured vs ${fmt(sp.coast.model_at_2ms, 3)} plant (m/s²)` : undefined],
    ]);

    const steerTiles = h(
        "div",
        { class: "tiles" },
        tile("Max left", fmt(st.max_left, 2), "", null, "normalised command"),
        tile("Max right", fmt(st.max_right, 2), ""),
        tile("Saturated", fmt(st.saturated_pct, 1), "%", st.saturated_pct > 5 ? "warn" : null),
        tile("Rate RMS", fmt(st.rate_rms_per_s, 2), "/s"),
        tile("Jerk (mean Δ²)", fmt(st.jerk_ms, 5), ""),
        tile("Reversals", fmt(st.reversals_per_s, 2), "/s", st.reversals_per_s > 3 ? "warn" : null),
        tile("Left / right time", `${fmt(st.left_share_pct, 0)} / ${fmt(st.right_share_pct, 0)}`, "%"),
        tile("Authority L / R", st.gain_left !== undefined ? `${fmt(st.gain_left, 2)} / ${fmt(st.gain_right, 2)}` : null, "× plant", st.gain_left !== undefined && (Math.abs(st.gain_left - 1) > 0.15 || Math.abs(st.gain_right - 1) > 0.15) ? "warn" : "good"),
    );

    const lagCard = lag
        ? h(
              "div",
              { class: "stack", style: { gap: "10px" } },
              h(
                  "div",
                  { class: "row" },
                  lag.at_search_bound ? statusBadge("warn", "fit at search bound") : Math.abs(lag.measured_total_s - lag.model_total_s) <= 0.15 ? statusBadge("good", "matches the plant") : statusBadge("warn", "differs from the plant"),
              ),
              kv([
                  ["Measured delay + lag", `${fmt(lag.dead_time_s, 2)} s + ${fmt(lag.tau_s, 2)} s = ${fmt(lag.measured_total_s, 2)} s`],
                  ["Plant", `${fmt(lag.model_dead_time_s, 2)} dead + ${fmt(lag.model_servo_tau_s, 2)} servo + ${fmt(lag.model_tau_s, 2)} chassis = ${fmt(lag.model_total_s, 2)} s`],
                  ["Gain left / right", `${fmt(lag.gain_left, 3)} / ${fmt(lag.gain_right, 3)} × plant`],
                  ["Fit error", `${fmt(lag.relative_error, 3)} (no-lag model: ${fmt(lag.relative_error_no_lag, 3)})`],
                  ["Samples", lag.samples],
              ]),
              h(
                  "div",
                  { class: "note" },
                  "Fitted over every moving sample: measured yaw rate ≈ first-order lag of (gain × the plant's kinematic yaw rate for the recorded command and speed).  Checked against plant.py driven by a recorded command stream, where it recovers the total delay within 0.05 s.  A mismatch here moves every turn-in the policy makes: its steering prior predicts through yaw_response_tau.",
              ),
          )
        : h("div", { class: "muted" }, "Not enough turning at speed to fit.");

    const auth = st.authority || [];
    const authBody = h("div");

    root.append(
        h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "Vehicle model"), h("p", {}, "How the car actually behaved, measured against plant.py and vehicle.yaml: the model the policy was trained on.  Differences here are sim-to-real gaps, not policy faults."))),
        h("div", { class: "stack" }, h("h3", { style: { margin: "0" } }, "Speed"), speedTiles),
        h("div", { class: "grid cols-2", style: { marginTop: "16px" } }, card("Speed loop & drivetrain", null, trackingKv), card("Speed", "measured vs commanded vs cap", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "speed", label: "measured", color: "--series-1", width: 2 }, { key: "cmd_speed", label: "commanded", color: "--series-2" }, { key: "v_cap", label: "cap", color: "--text-muted" }], { unit: "m/s", bands: lapBands(summary), height: 200 })); return b; })())),
        h("div", { class: "stack", style: { marginTop: "24px" } }, h("h3", { style: { margin: "0" } }, "Steering"), steerTiles),
        h(
            "div",
            { class: "grid cols-2", style: { marginTop: "16px" } },
            card("Steering → yaw response", "the chassis-lag check", lagCard),
            card("Steady-state authority", "curvature per command in held corners, measured vs plant", authBody),
        ),
        h("div", { style: { marginTop: "16px" } }, card("Measured vs plant-predicted yaw rate", "where these two part, the car is not the car the policy trained on", (() => { const b = h("div"); charts.push(timeChart(b, series, [{ key: "yaw_rate", label: "measured", color: "--series-1", width: 2 }, { key: "yaw_rate_plant", label: "plant prediction", color: "--series-2" }], { unit: "rad/s", bands: lapBands(summary), height: 220 })); return b; })())),
    );

    if (auth.length) {
        charts.push(
            xyChart(
                authBody,
                [
                    { label: "measured", color: "--series-1", x: auth.map((a) => a.cmd), y: auth.map((a) => a.kappa_measured), points: true },
                    { label: "plant", color: "--series-2", x: auth.map((a) => a.cmd), y: auth.map((a) => a.kappa_plant) },
                ],
                { xlabel: "steering command", ylabel: "curvature 1/m", height: 200 },
            ),
        );
        authBody.append(table([{ label: "Command", key: "cmd", num: true }, { label: "Samples", key: "n", num: true }, { label: "κ measured", num: true, get: (a) => fmt(a.kappa_measured, 3) }, { label: "κ plant", num: true, get: (a) => fmt(a.kappa_plant, 3) }], auth));
    } else authBody.append(h("div", { class: "muted" }, `Only ${st.steady_samples ?? 0} steady-corner samples; the dynamic fit on the left uses every sample instead.`));

    return () => charts.forEach((c) => c && c.destroy());
}
