// Watch the run back in Rerun -- the course and car in 3D, the ZED clouds and
// accumulated map, the camera, every channel and the logs, all on the Run
// Lab's clock.

import { api } from "../api.js";
import { h, icon, toast } from "../ui.js";
import { RerunView } from "../rerunview.js";

export default async function replayPage(root, { run, summary }) {
    const per = summary.perception || {};

    const native = async (which) => {
        try {
            await api.post(`/api/runs/${encodeURIComponent(run)}/rerun`, { which });
            toast(which === "bag" ? "Opening the raw bag in the Rerun viewer…" : "Opening the Rerun viewer…");
        } catch (e) {
            toast(e.message, "bad");
        }
    };
    const nativeBtn = h("button", { class: "btn sm" }, icon("external"), "Open in Rerun app");
    nativeBtn.addEventListener("click", () => native("recording"));
    const bagBtn = h("button", { class: "btn sm", title: "Every topic in the bag, as recorded (map frame, no analysis)" }, icon("external"), "Raw bag in Rerun");
    bagBtn.addEventListener("click", () => native("bag"));
    const followCb = h("input", { type: "checkbox" });
    const holder = h("div");

    const contents = [
        per.fused_points ? `ZED map ${per.fused_points.toLocaleString()} pts` : null,
        per.clouds ? `${per.clouds} cloud frames, accumulated map ${(per.map_points || 0).toLocaleString()} pts` : null,
        !per.fused_points && !per.clouds ? "no map or point cloud" : null,
        per.images ? `${per.images} camera frames` : "no camera",
    ].filter(Boolean);

    // No page header: every pixel goes to the viewer.
    root.append(
        h(
            "section",
            { class: "card" },
            h(
                "div",
                { class: "card-head", title: "The course and car in 3D (Course, or Follow car to ride along), the ZED clouds and the map they build, the camera, every channel and the logs.  Its clock and the CfR Log Playback's timeline are one: scrub either." },
                h("h3", {}, "Replay"),
                h("span", { class: "sub" }, contents.join(" · ")),
                h("span", { class: "spacer" }),
                h("label", { class: "check small", title: "Open the 3D view that tracks the car (the Follow car tab)" }, followCb, "Follow car"),
                nativeBtn,
                bagBtn,
                h("a", { class: "btn sm", href: `/api/runs/${encodeURIComponent(run)}/recording.rrd`, download: `${run}.rrd` }, icon("download"), ".rrd"),
            ),
            h("div", { class: "card-body flush", style: { paddingTop: "10px" } }, holder),
        ),
    );

    const viewer = new RerunView(holder, { run, stamp: (summary.meta || {}).processed_utc || "" });
    followCb.addEventListener("change", () => viewer.follow(followCb.checked));
    return () => viewer.destroy();
}
