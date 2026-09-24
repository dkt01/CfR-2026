// Watch the run back in Rerun -- the course and car in 3D, the ZED clouds and
// accumulated map, the camera, every channel and the logs, all on the Run
// Lab's clock -- and, when this laptop has ROS, replay it in Gazebo or RViz.

import { api } from "../api.js";
import { h, icon, card, fmt, toast, statusBadge, kv } from "../ui.js";
import { playhead } from "../playhead.js";
import { RerunView } from "../rerunview.js";

export default async function replayPage(root, { run, summary }) {
    const per = summary.perception || {};

    // ------------------------------------------------------------- rerun
    const native = async (which) => {
        try {
            await api.post(`/api/runs/${encodeURIComponent(run)}/rerun`, { which });
            toast(which === "bag" ? "Opening the raw bag in the Rerun viewer…" : "Opening the Rerun viewer…");
        } catch (e) {
            toast(e.message, "bad");
        }
    };
    const nativeBtn = h("button", { class: "btn" }, icon("external"), "Open in Rerun app");
    nativeBtn.addEventListener("click", () => native("recording"));
    const bagBtn = h("button", { class: "btn", title: "Every topic in the bag, as recorded (map frame, no analysis)" }, icon("external"), "Raw bag in Rerun");
    bagBtn.addEventListener("click", () => native("bag"));
    const holder = h("div");

    // ------------------------------------------------------- external replay
    const ext = h("div");
    const rateIn = h("select", {}, [0.25, 0.5, 1, 2, 4].map((r) => h("option", { value: r, selected: r === 1 }, `${r}×`)));
    const guiCb = h("input", { type: "checkbox" });
    const webCb = h("input", { type: "checkbox", checked: true });
    const loopCb = h("input", { type: "checkbox" });
    const fromPlayhead = h("input", { type: "checkbox", checked: true });
    let pollTimer = null;
    let alive = true;

    async function startGazebo() {
        try {
            await api.post("/api/replay/gazebo", { run, gui: guiCb.checked, web: webCb.checked, rate: Number(rateIn.value), start: fromPlayhead.checked ? playhead.t : summary.window.t0, loop: loopCb.checked });
            toast("Starting Gazebo… the world takes ~15 s to come up");
        } catch (e) {
            toast(e.message, "bad");
        }
        pollExt();
    }
    async function startRviz() {
        try {
            await api.post("/api/replay/rviz", { run, rate: Number(rateIn.value), start: fromPlayhead.checked ? Math.max(0, playhead.t) : 0, loop: loopCb.checked });
            toast("Playing the bag into RViz");
        } catch (e) {
            toast(e.message, "bad");
        }
        pollExt();
    }
    async function control(action, body = {}) {
        try {
            await api.post("/api/replay/control", { action, ...body });
        } catch (e) {
            toast(e.message, "bad");
        }
    }
    async function pollExt() {
        clearTimeout(pollTimer);
        let s;
        try {
            s = await api.get("/api/replay");
        } catch {
            return;
        }
        if (!alive) return;
        renderExt(s);
        if (s.active) pollTimer = setTimeout(pollExt, 1500);
    }

    function renderExt(s) {
        const gz = h("button", { class: "btn primary", disabled: !s.ros_available }, icon("gazebo"), "Replay in Gazebo");
        gz.addEventListener("click", startGazebo);
        const rv = h("button", { class: "btn", disabled: !s.ros_available }, icon("replay"), "Play bag in RViz");
        rv.addEventListener("click", startRviz);
        const stop = h("button", { class: "btn danger", disabled: !s.active }, icon("x"), "Stop");
        stop.addEventListener("click", async () => {
            await api.post("/api/replay/stop");
            pollExt();
        });
        const rows = [
            h("div", { class: "row" }, gz, rv, stop),
            h("div", { class: "row small" }, h("span", { class: "muted" }, "rate"), rateIn, h("label", { class: "check" }, fromPlayhead, "start at playhead"), h("label", { class: "check" }, loopCb, "loop"), h("label", { class: "check" }, guiCb, "Gazebo window"), h("label", { class: "check" }, webCb, "web viewer")),
        ];
        if (!s.ros_available) rows.push(h("div", { class: "note warn" }, "ROS 2 Jazzy is not installed on this machine, so Gazebo and RViz replays are unavailable.  Rerun above needs nothing."));
        if (s.active) {
            rows.push(h("div", { class: "row" }, statusBadge("good", `${s.mode} replay running`), h("span", { class: "small muted" }, `${s.run} · ROS_DOMAIN_ID=${s.domain} · GZ_PARTITION=${s.partition}`)));
            if (s.mode === "gazebo") {
                const links = h("div", { class: "row" });
                if (s.viewer_url) links.append(h("a", { class: "btn", href: s.viewer_url, target: "_blank" }, icon("external"), s.viewer ? "Open web viewer" : "Web viewer (starting…)"));
                const gui = h("button", { class: "btn" }, icon("external"), "Open Gazebo window");
                gui.addEventListener("click", async () => {
                    try {
                        await api.post("/api/replay/gui");
                        toast("Gazebo GUI launching on this machine's display");
                    } catch (e) {
                        toast(e.message, "bad");
                    }
                });
                const sync = h("button", { class: "btn" }, "Sync Gazebo to playhead");
                sync.addEventListener("click", () => control("seek", { t: playhead.t }));
                const pp = h("button", { class: "btn" }, s.ghost?.playing ? "Pause Gazebo" : "Play Gazebo");
                pp.addEventListener("click", () => control(s.ghost?.playing ? "pause" : "play").then(pollExt));
                links.append(gui, sync, pp);
                rows.push(links);
                rows.push(
                    kv([
                        ["Gazebo clock", s.ghost ? `${fmt(s.ghost.t, 1)} s of ${fmt(s.ghost.t_end, 1)} s` : "waiting for the world…"],
                        ["Poses sent", s.ghost ? `${s.ghost.sent} (${s.ghost.dropped} dropped while Gazebo was busy)` : "—"],
                        ["set_pose service", s.ghost ? (s.ghost.service_ready ? "ready" : "not up yet") : "—"],
                        ["Native GUI", h("code", { style: { fontSize: "11px" } }, `GZ_PARTITION=${s.partition} gz sim -g`)],
                    ]),
                );
                if (s.websocket_note) rows.push(h("div", { class: "note warn" }, s.websocket_note));
                if (s.viewer_note) rows.push(h("div", { class: "note" }, s.viewer_note));
            }
            for (const [name, p] of Object.entries(s.procs || {})) {
                if (!p.alive && p.log.length) rows.push(h("details", {}, h("summary", { class: "small" }, `${name} exited — log`), h("pre", { class: "mono small", style: { whiteSpace: "pre-wrap" } }, p.log.join("\n"))));
            }
        }
        ext.replaceChildren(h("div", { class: "stack", style: { gap: "10px" } }, rows));
    }

    const contents = [
        per.map_points ? `accumulated map ${per.map_points.toLocaleString()} pts` : "no point cloud",
        per.clouds ? `${per.clouds} cloud frames` : null,
        per.fused_points ? "ZED spatial map" : null,
        per.images ? `${per.images} camera frames` : "no camera",
    ].filter(Boolean);

    root.append(
        h(
            "div",
            { class: "page-head" },
            h("div", {}, h("h2", {}, "Replay"), h("p", {}, "The run in Rerun: the course and car in 3D (Course, or Chase to ride along), the ZED clouds and the map they build, the camera, every channel and the logs.  Its clock and the CfR Log Playback's timeline are one: scrub either.")),
            h("span", { class: "spacer" }),
            nativeBtn,
            bagBtn,
        ),
        h("section", { class: "card" }, h("div", { class: "card-head" }, h("h3", {}, "Rerun"), h("span", { class: "sub" }, contents.join(" · ")), h("span", { class: "spacer" }), h("a", { class: "btn sm", href: `/api/runs/${encodeURIComponent(run)}/recording.rrd`, download: `${run}.rrd` }, icon("download"), ".rrd")), h("div", { class: "card-body flush", style: { paddingTop: "10px" } }, holder)),
        h("div", { style: { marginTop: "16px" } }, card("Replay in Gazebo or RViz", "the car follows its recorded path in the simulator, or the bag plays into a live ROS graph", ext)),
    );

    const viewer = new RerunView(holder, { run, height: Math.max(560, window.innerHeight - 330) });
    pollExt();
    return () => {
        alive = false;
        clearTimeout(pollTimer);
        viewer.destroy();
    };
}
