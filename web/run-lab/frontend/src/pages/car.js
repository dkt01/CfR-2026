// The Orin, over USB-C (or Wi-Fi): what is on it, what is recording, and a
// one-click pull that verifies the copy and queues processing.

import { api } from "../api.js";
import { h, icon, card, bytes, when, statusBadge, toast, table, empty, fmt } from "../ui.js";
import { jobs } from "../jobs.js";

const DRIVER_STATE = { 0: "waiting for start", 1: "racing", 2: "coasting to stop", 3: "finished", 4: "POSE STALE", 5: "manual stop" };

export default async function carPage(root, { go, refreshRuns }) {
    const health = await api.get("/api/health");
    let settings = health.settings;
    let timer = null;
    let alive = true;

    const hostIn = h("input", { type: "text", value: settings.host, style: { width: "240px" } });
    const remoteIn = h("input", { type: "text", value: settings.remote, style: { width: "160px" } });
    const save = h("button", { class: "btn" }, "Save");
    save.addEventListener("click", async () => {
        settings = await api.post("/api/settings", { host: hostIn.value, remote: remoteIn.value });
        toast("Saved");
        load();
    });
    const refresh = h("button", { class: "btn" }, icon("refresh"), "Refresh");
    refresh.addEventListener("click", () => load());

    const statusCard = h("div");
    const listCard = h("div");
    root.replaceChildren(
        h(
            "div",
            { class: "page-head" },
            h("div", {}, h("h2", {}, "Car (Orin)"), h("p", {}, "Runs are recorded on the Orin into ~/cfr_runs.  Plug the Orin into this laptop over USB-C (it appears as 192.168.55.1) and pull them here.  Nothing is ever deleted from the car unless you ask, and only after the copy here has been verified byte-for-byte.")),
            h("span", { class: "spacer" }),
            refresh,
        ),
        h("div", { class: "stack" }, card("Connection", null, h("div", { class: "row" }, h("span", { class: "muted" }, "SSH host"), hostIn, h("span", { class: "muted" }, "runs dir"), remoteIn, save)), statusCard, listCard),
    );

    async function load() {
        statusCard.replaceChildren(card(null, null, h("div", { class: "muted" }, "Contacting the Orin…")));
        let s;
        try {
            s = await api.get("/api/orin");
        } catch (e) {
            statusCard.replaceChildren(card(null, null, h("div", { class: "note bad" }, e.message)));
            return;
        }
        if (!alive) return;
        renderStatus(s);
        if (s.reachable) renderList(s);
        else listCard.replaceChildren();
        clearTimeout(timer);
        // While a run records, keep its live status fresh.
        if ((s.runs || []).some((r) => r.recording)) timer = setTimeout(load, 3000);
    }

    function renderStatus(s) {
        const usb = s.usb_link === true ? statusBadge("good", "USB-C link up") : s.usb_link === false ? statusBadge("info", "no USB-C link") : null;
        if (!s.reachable) {
            statusCard.replaceChildren(
                card(
                    "Not reachable",
                    s.host,
                    h(
                        "div",
                        { class: "stack", style: { gap: "10px" } },
                        h("div", { class: "row" }, usb, statusBadge("bad", s.error || "unreachable")),
                        s.hint ? h("div", { class: "note warn" }, s.hint) : null,
                        h(
                            "div",
                            { class: "note" },
                            "Checklist: the Orin is powered and booted; the USB-C cable is in the Orin's device port; this laptop shows a 192.168.55.x address (",
                            h("code", {}, "ip -4 addr"),
                            "); key-based ssh works (",
                            h("code", {}, `ssh ${s.host} true`),
                            " returns without a password prompt).",
                        ),
                    ),
                ),
            );
            return;
        }
        const used = s.total_bytes ? 1 - s.free_bytes / s.total_bytes : 0;
        const recording = (s.runs || []).filter((r) => r.recording);
        statusCard.replaceChildren(
            h(
                "div",
                { class: "grid cols-3" },
                card(
                    "Connected",
                    s.host,
                    h("div", { class: "stack", style: { gap: "8px" } }, h("div", { class: "row" }, usb, statusBadge("good", s.host)), h("div", { class: "muted small" }, `${(s.runs || []).length} runs in ${s.root}`)),
                ),
                card(
                    "Orin disk",
                    null,
                    h(
                        "div",
                        {},
                        h("div", { style: { fontSize: "20px", fontWeight: 650 } }, `${bytes(s.free_bytes)} free`),
                        h("div", { class: "progress", style: { marginTop: "8px" } }, h("div", { style: { width: `${used * 100}%`, background: used > 0.9 ? "var(--critical)" : used > 0.75 ? "var(--warning)" : "var(--accent)" } })),
                        h("div", { class: "small muted", style: { marginTop: "4px" } }, `${Math.round(used * 100)}% of ${bytes(s.total_bytes)} used · the recorder refuses to start below 3 GB`),
                    ),
                ),
                card(
                    "Recording now",
                    null,
                    recording.length
                        ? recording.map((r) => liveStatus(r))
                        : h("div", { class: "muted" }, "Nothing is recording.  Runs record automatically when formula_one.launch.py is started with use_sim_time:=false."),
                ),
            ),
        );
    }

    function liveStatus(r) {
        const st = r.status || {};
        return h(
            "div",
            { class: "stack", style: { gap: "6px" } },
            h("div", { class: "row" }, statusBadge("bad", "REC"), h("b", {}, r.meta.label || r.name)),
            h(
                "div",
                { class: "small" },
                `${fmt(st.elapsed_s, 0)} s · bag ${bytes(st.bag_bytes)} · ${DRIVER_STATE[st.driver_state] ?? "driver not seen"}`,
                st.lap !== null && st.lap !== undefined ? ` · lap ${st.lap}/${st.laps_target}` : "",
                st.speed !== null && st.speed !== undefined ? ` · ${fmt(st.speed, 2)} m/s` : "",
                st.min_clearance !== null && st.min_clearance !== undefined ? ` · min clearance ${fmt(st.min_clearance, 3)} m` : "",
            ),
            st.bag_alive === false ? h("div", { class: "note bad" }, "ros2 bag record is not running") : null,
        );
    }

    function renderList(s) {
        const runs = s.runs || [];
        if (!runs.length) {
            listCard.replaceChildren(card("Runs on the car", null, empty("No runs on the Orin yet", "Drive with the recorder on and they appear here.")));
            return;
        }
        const rows = runs.map((r) => ({ ...r, job: jobs.active("pull", r.name) }));
        const cols = [
            { label: "Run", get: (r) => h("div", {}, h("div", { style: { fontWeight: 600 } }, r.meta.label || r.meta.profile || r.name), h("div", { class: "small muted mono" }, r.name)) },
            { label: "Kind", get: (r) => h("span", { class: "badge" }, r.meta.kind || (r.meta.profile ? "characterization" : "drive")) },
            { label: "Started", get: (r) => when(r.meta.started_utc) },
            { label: "Size", num: true, get: (r) => bytes(r.bytes) },
            {
                label: "State",
                get: (r) =>
                    r.recording
                        ? statusBadge("bad", "recording")
                        : r.pulled?.verified
                          ? statusBadge("good", "on laptop · verified")
                          : r.local
                            ? statusBadge("warn", "partial copy")
                            : h("span", { class: "badge" }, "on car only"),
            },
            { label: "", get: (r) => actions(r) },
        ];
        listCard.replaceChildren(card("Runs on the car", `${runs.length} runs · ${bytes(runs.reduce((a, r) => a + r.bytes, 0))}`, table(cols, rows)));
    }

    function actions(r) {
        const wrap = h("div", { class: "row", style: { justifyContent: "flex-end" } });
        const job = jobs.active("pull", r.name);
        if (job) {
            wrap.append(h("div", { style: { width: "180px" } }, h("div", { class: "small muted" }, job.message || "starting…"), h("div", { class: "progress" }, h("div", { style: { width: `${job.progress * 100}%` } }))));
            return wrap;
        }
        const pull = h("button", { class: `btn sm ${r.local ? "" : "primary"}`, disabled: r.recording }, icon("download"), r.local ? "Pull again" : "Pull");
        pull.addEventListener("click", async () => {
            await api.post("/api/orin/pull", { run: r.name, process: true });
            toast(`Pulling ${r.name}; it will be processed when the copy is verified`);
            jobs.poll();
            load();
        });
        wrap.append(pull);
        if (r.local) {
            const open = h("button", { class: "btn sm" }, "Open");
            open.addEventListener("click", () => go(`#/run/${encodeURIComponent(r.name)}/overview`));
            wrap.append(open);
        }
        if (r.pulled?.verified && !r.recording) {
            const del = h("button", { class: "btn sm danger", title: "Delete from the Orin (the laptop copy is kept)" }, icon("trash"));
            del.addEventListener("click", async () => {
                if (!confirm(`Delete ${r.name} from the Orin?\n\nThe verified copy on this laptop is kept.  This cannot be undone on the car.`)) return;
                try {
                    await api.post("/api/orin/delete", { run: r.name });
                    toast(`Deleted ${r.name} from the Orin`, "good");
                } catch (e) {
                    toast(e.message, "bad");
                }
                load();
            });
            wrap.append(del);
        }
        return wrap;
    }

    const off = jobs.onChange((changed) => {
        if (changed.some((j) => j.kind === "pull")) {
            for (const j of changed) {
                if (j.kind === "pull" && j.state === "done") {
                    toast(`Pulled ${j.target}${j.result?.verified ? " (verified)" : " — size check did not match; pull again"}`, j.result?.verified ? "good" : "bad");
                    refreshRuns();
                    load();
                } else if (j.kind === "pull" && j.state === "failed") toast(`Pull failed: ${j.error}`, "bad");
            }
            if (alive) api.get("/api/orin").then((s) => alive && renderList(s)).catch(() => {});
        }
    });

    load();
    return () => {
        alive = false;
        clearTimeout(timer);
        off();
    };
}
