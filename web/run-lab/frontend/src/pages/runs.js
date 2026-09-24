// The run library on this laptop.

import { api } from "../api.js";
import { h, icon, card, when, statusBadge, toast, empty, fmt } from "../ui.js";
import { jobs } from "../jobs.js";

export default async function runsPage(root, { go, refreshRuns }) {
    let filter = "";
    const runs = await refreshRuns();
    const search = h("input", { type: "text", placeholder: "Filter runs…", style: { width: "240px" } });
    search.addEventListener("input", () => {
        filter = search.value.toLowerCase();
        draw();
    });
    const importIn = h("input", { type: "text", placeholder: "/path/to/a/run/folder", style: { width: "280px" } });
    const importBtn = h("button", { class: "btn" }, "Link folder");
    importBtn.addEventListener("click", async () => {
        try {
            await api.post("/api/runs/import", { path: importIn.value });
            toast("Linked", "good");
            await refreshRuns();
            draw();
        } catch (e) {
            toast(e.message, "bad");
        }
    });
    const list = h("div", { class: "run-list" });
    const health = await api.get("/api/health");

    root.replaceChildren(
        h(
            "div",
            { class: "page-head" },
            h("div", {}, h("h2", {}, "Runs"), h("p", {}, `Runs on this laptop, in ${health.runs_dir}.  Pull new ones from the Car page, or link a folder someone copied over.`)),
            h("span", { class: "spacer" }),
            search,
        ),
        list,
        h("div", { style: { marginTop: "18px" } }, card("Add a run from elsewhere", "a USB stick, sync_runs.sh output, a teammate's copy", h("div", { class: "row" }, importIn, importBtn))),
    );

    function draw() {
        const shown = runs.filter((r) => !filter || `${r.name} ${r.label} ${r.driver || ""}`.toLowerCase().includes(filter));
        if (!shown.length) {
            list.replaceChildren(
                card(null, null, empty(runs.length ? "No runs match" : "No runs yet", runs.length ? "Clear the filter." : "Connect the Orin and pull a run from the Car page.", runs.length ? null : h("a", { class: "btn primary", href: "#/car" }, icon("usb"), "Go to Car"))),
            );
            return;
        }
        list.replaceChildren(...shown.map(runCard));
    }

    function runCard(r) {
        const k = Object.fromEntries((r.kpis || []).map((x) => [x.key, x]));
        const job = jobs.active("process", r.name);
        const state = job
            ? h("div", { style: { width: "160px" } }, h("div", { class: "small muted" }, job.message), h("div", { class: "progress" }, h("div", { style: { width: `${job.progress * 100}%` } })))
            : r.recording
              ? statusBadge("warn", "incomplete: was recording")
              : !r.has_bag
                ? h("span", { class: "badge" }, "no bag")
                : !r.processed
                  ? statusBadge("info", "not processed")
                  : k.result
                    ? statusBadge(k.result.tone === "good" ? "good" : "bad", k.result.value)
                    : null;
        const del = h("button", { class: "btn sm ghost", title: "Delete from this laptop" }, icon("trash"));
        del.addEventListener("click", async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete ${r.name} from this laptop?\n\n${r.pulled?.verified ? "It was verified on the Orin when pulled; check it is still there if you need it." : "This may be the only copy."}`)) return;
            await api.del(`/api/runs/${encodeURIComponent(r.name)}`);
            toast(`Deleted ${r.name}`);
            await refreshRuns();
            runs.splice(runs.indexOf(r), 1);
            draw();
        });
        const el = h(
            "div",
            { class: "card run-card" },
            h(
                "div",
                {},
                h("div", { class: "row" }, h("span", { class: "name" }, r.label), r.simulation ? h("span", { class: "badge info" }, "simulation") : null, h("span", { class: "badge" }, r.kind)),
                h("div", { class: "meta" }, h("span", { class: "mono" }, r.name), r.started_utc ? h("span", {}, when(r.started_utc)) : null, r.driver ? h("span", {}, `driver ${r.driver}`) : null, r.speed_scale ? h("span", {}, `speed ×${r.speed_scale}`) : null),
                r.processed
                    ? h(
                          "div",
                          { class: "kpis" },
                          ["race_time", "max_speed", "min_clear", "cte", "jumps"].map((key) =>
                              k[key] ? h("span", {}, h("span", { class: "muted" }, `${k[key].label} `), h("b", {}, fmt(k[key].value, key === "jumps" ? 0 : 2)), k[key].unit ? ` ${k[key].unit}` : "") : null,
                          ),
                          r.verdict_counts ? h("span", {}, h("span", { class: "muted" }, "verdicts "), h("b", {}, `${r.verdict_counts.worked} ✓ · ${r.verdict_counts.failed} ✗`)) : null,
                      )
                    : null,
            ),
            h("div", { class: "row" }, state, del),
        );
        el.addEventListener("click", () => go(`#/run/${encodeURIComponent(r.name)}/overview`));
        return el;
    }

    draw();
    const off = jobs.onChange(() => draw());
    return () => off();
}
