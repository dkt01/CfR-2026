// The verdict first: did it work, what did not, and where.

import { h, icon, card, tile, fmt, clock, table, statusBadge, kv, when } from "../ui.js";
import { seek, subscribe, playhead } from "../playhead.js";
import { TrackMap } from "../trackmap.js";

const TONE = { good: "good", warn: "warn", bad: "bad" };
const KIND_ICON = { lap: "check", start: "play", finish: "check", contact: "x", graze: "warn", stall: "x", overspeed: "warn", pose: "warn", link: "x", estop: "x", mode: "info", log: "x", stop: "info" };

export default async function overviewPage(root, { run, data, summary, series, go }) {
    const course = await data.course();
    const meta = summary.meta || {};
    const md = meta.metadata || {};

    const tiles = h(
        "div",
        { class: "tiles" },
        summary.kpis.map((k) => tile(k.label, typeof k.value === "number" ? fmt(k.value, ["jumps"].includes(k.key) ? 0 : ["max_speed", "race_time", "duration"].includes(k.key) ? 2 : 3) : k.value, k.unit, TONE[k.tone])),
    );

    const verdictList = (items, emptyText) =>
        items.length
            ? h(
                  "div",
                  {},
                  items.map((v) => {
                      const el = h(
                          "div",
                          { class: "verdict", "data-t": v.t ?? null },
                          statusBadge(v.severity === "good" ? "good" : v.severity === "warn" ? "warn" : "bad", ""),
                          h("div", {}, h("div", { class: "title" }, v.title), h("div", { class: "detail" }, v.detail)),
                          v.t !== null && v.t !== undefined ? h("span", { class: "jump" }, `${clock(v.t)} →`) : null,
                      );
                      if (v.t !== null && v.t !== undefined) el.addEventListener("click", () => seek(v.t, "verdict"));
                      return el;
                  }),
              )
            : h("div", { class: "muted", style: { padding: "10px 12px" } }, emptyText);

    const verdicts = h(
        "div",
        { class: "verdicts" },
        card("What worked", `${summary.verdicts.worked.length}`, verdictList(summary.verdicts.worked, "Nothing stood out as working.")),
        card("What did not", `${summary.verdicts.failed.length}`, verdictList(summary.verdicts.failed, "No problems found.")),
    );

    const mapHolder = h("div");
    const mapCard = h(
        "section",
        { class: "card" },
        h("div", { class: "card-head" }, h("h3", {}, "Where it happened"), h("span", { class: "sub" }, "path coloured by clearance · click the path or any verdict to jump"), h("span", { class: "spacer" }), h("a", { class: "btn sm", href: `#/run/${encodeURIComponent(run)}/track` }, "Open track view")),
        h("div", { class: "card-body flush", style: { paddingTop: "10px" } }, mapHolder),
    );

    const laps = (summary.laps || {}).laps || [];
    const lapTable = laps.length
        ? table(
              [
                  { label: "Lap", key: "lap" },
                  { label: "Time", num: true, get: (l) => `${fmt(l.time, 2)} s` },
                  { label: "Δ", num: true, get: (l) => (l.delta === undefined ? "—" : h("span", { style: { color: l.delta < 0 ? "var(--good-ink)" : "var(--critical-ink)" } }, `${l.delta > 0 ? "+" : ""}${fmt(l.delta, 2)}`)) },
                  { label: "Top speed", num: true, get: (l) => `${fmt(l.max_speed, 2)} m/s` },
                  { label: "Avg", num: true, get: (l) => `${fmt(l.mean_speed, 2)} m/s` },
                  { label: "Min clear", num: true, get: (l) => `${fmt(l.min_clearance, 3)} m` },
                  { label: "|CTE|", num: true, get: (l) => `${fmt(l.mean_abs_cte, 3)} m` },
              ],
              laps,
              (l) => seek(l.t0, "lap"),
          )
        : h("div", { class: "muted" }, "No complete laps.");

    const events = summary.events || [];
    const evList = h("div", { class: "events" });
    const evRows = events.map((e) => {
        const row = h(
            "div",
            { class: "event" },
            h("span", { class: "t" }, clock(e.t)),
            h("span", { style: { color: e.severity === "bad" ? "var(--critical)" : e.severity === "warn" ? "var(--warning-ink)" : e.severity === "good" ? "var(--good)" : "var(--text-muted)" } }, icon(KIND_ICON[e.kind] || "info", "status-icon")),
            h("span", {}, e.text),
            h("span", { class: "st" }, e.station !== null && e.station !== undefined ? `${fmt(e.station, 1)} m` : ""),
        );
        row.addEventListener("click", () => seek(e.t, "event"));
        return row;
    });
    evList.append(...evRows);
    const kindCounts = {};
    events.forEach((e) => (kindCounts[e.kind] = (kindCounts[e.kind] || 0) + 1));

    const about = kv([
        ["Label", md.label || run],
        ["Started", md.started_utc ? when(md.started_utc) : "—"],
        ["Source", meta.simulation ? "Gazebo simulation" : "the car"],
        ["Driver", meta.driver || md.driver || "—"],
        ["Policy", md.policy ? h("span", { class: "mono", title: md.policy.path }, `${md.policy.path.split("/").slice(-2).join("/")} · ${md.policy.sha256.slice(0, 10)}`) : "—"],
        ["Speed scale", md.speed_scale || "—"],
        ["Git", md.git_sha ? h("span", { class: "mono" }, md.git_sha.slice(0, 12)) : "—"],
        ["Track frame", h("span", {}, summary.frame.method, summary.frame.fit_rms_m ? ` (fit RMS ${fmt(summary.frame.fit_rms_m, 3)} m)` : "")],
        ["Run window", `${clock(summary.window.t0)} – ${clock(summary.window.t1)} (${summary.window.source})`],
        ["Notes", md.notes || undefined],
    ]);

    root.append(
        h("div", { class: "stack" }, tiles, verdicts),
        h(
            "div",
            { class: "split", style: { marginTop: "16px" } },
            mapCard,
            h(
                "div",
                { class: "stack" },
                card("Laps", null, lapTable),
                card(
                    "Events",
                    `${events.length} · ${Object.entries(kindCounts).map(([k, n]) => `${n} ${k}`).join(", ")}`,
                    evList,
                ),
            ),
        ),
        h("div", { class: "grid cols-2", style: { marginTop: "16px" } }, card("About this run", null, about), card("Processing", null, kv([["Processed", when(meta.processed_utc)], ["Took", `${meta.processing_s} s`], ["Bag", `${fmt(summary.bag.duration_s, 1)} s · ${Object.keys(summary.bag.topics).length} topics`], ["Analysis version", meta.version]]))),
    );

    const map = new TrackMap(mapHolder, { course, series, summary, metric: "clearance", height: 460 });
    // Highlight the event nearest the playhead.
    const off = subscribe((t) => {
        let best = -1;
        let bd = 0.6;
        events.forEach((e, i) => {
            const d = Math.abs(e.t - t);
            if (d < bd) {
                bd = d;
                best = i;
            }
        });
        evRows.forEach((r, i) => (r.style.background = i === best ? "var(--accent-soft)" : ""));
    });
    void playhead;
    return () => {
        map.destroy();
        off();
    };
}
