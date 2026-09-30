// Live ZED exposure/ROI tuning while connected to the car. Needs
// zed_live_tuning.launch.py running there already (rosbridge + web_video_server
// -- camera-only, arms nothing). Sliders push `ros2 param set` immediately for
// visual feedback; "Save" writes the finished values into
// jetson/cfr_arduino_bridge/config/cfr_zed2i.yaml on this laptop, the one copy
// (zed/config/cfr_zed2i.yaml is a symlink to it) -- syncing that onto the car
// is the same deploy-to-car flow any other config edit uses.
//
// The ROI here is always the rectangle this file has ever used -- cut off the
// top by some fraction, keep the rest -- so this is a top-cut slider, not a
// general polygon editor; see cfr_zed2i.yaml's own comment on why the cut is
// where it is (sky, sun, canopy, spectators).

import { api } from "../api.js";
import { h, icon, card, toast, statusBadge } from "../ui.js";

function polygonForCut(cut) {
    return [[0.0, cut], [1.0, cut], [1.0, 1.0], [0.0, 1.0]];
}

export default async function zedTunePage(root) {
    const urls = await api.get("/api/zed/stream_urls");

    let auto = true;
    let exposure = 10;
    let gain = 20;
    let roiCut = 0.2;
    let debounceTimer = null;
    let statusTimer = null;
    let videoLoaded = false;

    const video = h("img", { class: "zed-stream", alt: "ZED rectified color" });
    const roiMask = h("img", { class: "zed-stream", alt: "ROI mask" });
    video.addEventListener("error", () => {
        if (videoLoaded) toast("Video stream dropped -- is zed_live_tuning.launch.py still running on the car?", "bad");
    });

    const connectionCard = h("div");
    const connectBtn = h("button", { class: "btn primary" }, icon("play"), "Connect");
    const stopBtn = h("button", { class: "btn danger" }, icon("x"), "Stop");
    connectBtn.addEventListener("click", async () => {
        connectBtn.disabled = true;
        try {
            const result = await api.post("/api/zed/connect", {});
            toast(result.already_running ? "Already running on the car" : "Starting zed_live_tuning.launch.py on the car…", result.ok ? "good" : "bad");
            if (!result.ok) toast(result.output || "Connect failed", "bad");
            pollStatus(8);
        } catch (e) {
            toast(e.message, "bad");
        } finally {
            connectBtn.disabled = false;
        }
    });
    stopBtn.addEventListener("click", async () => {
        stopBtn.disabled = true;
        try {
            const result = await api.post("/api/zed/disconnect", {});
            toast(result.ok ? "Stopped -- rosbridge and web_video_server are down, ready for an RL run or calibration" : "Stop may not have fully succeeded", result.ok ? "good" : "bad");
            // Drop the live connections client-side too, so a stale MJPEG
            // socket does not keep holding the topic open.
            video.removeAttribute("src");
            roiMask.removeAttribute("src");
            videoLoaded = false;
            pollStatus(5);
        } catch (e) {
            toast(e.message, "bad");
        } finally {
            stopBtn.disabled = false;
        }
    });

    function renderConnection(status) {
        const rosbridgeOk = status?.rosbridge;
        const videoOk = status?.video;
        connectionCard.replaceChildren(
            card(
                "Connection",
                null,
                h(
                    "div",
                    { class: "stack", style: { gap: "10px" } },
                    h(
                        "div",
                        { class: "row" },
                        statusBadge(rosbridgeOk ? "good" : "info", rosbridgeOk ? "rosbridge up" : "rosbridge not reachable"),
                        statusBadge(videoOk ? "good" : "info", videoOk ? "web_video_server up" : "web_video_server not reachable"),
                        h("span", { class: "spacer" }),
                        connectBtn,
                        stopBtn,
                    ),
                    h("div", { class: "note" }, "Connect starts zed_live_tuning.launch.py on the car over ssh if it is not already running (rosbridge + web_video_server -- camera-only, arms nothing). Needs the ZED itself already up (launch.sh, on the car). Stop it again before an RL run or a calibration profile, so it is not holding an extra subscriber on the camera."),
                ),
            ),
        );
        if (videoOk && !videoLoaded) {
            videoLoaded = true;
            video.src = urls.video;
            roiMask.src = urls.roi_mask;
        }
    }

    async function checkStatus() {
        try {
            const status = await api.get("/api/zed/status");
            renderConnection(status);
            return status;
        } catch {
            renderConnection(null);
            return null;
        }
    }

    function pollStatus(attemptsLeft) {
        clearTimeout(statusTimer);
        if (attemptsLeft <= 0) return;
        statusTimer = setTimeout(async () => {
            const status = await checkStatus();
            if (!status?.video || !status?.rosbridge) pollStatus(attemptsLeft - 1);
        }, 1000);
    }

    const autoToggle = h("input", { type: "checkbox", checked: true });
    const exposureSlider = h("input", { type: "range", min: "0", max: "100", value: String(exposure), disabled: true });
    const gainSlider = h("input", { type: "range", min: "0", max: "100", value: String(gain), disabled: true });
    const exposureLabel = h("span", { class: "mono small" }, String(exposure));
    const gainLabel = h("span", { class: "mono small" }, String(gain));
    const roiSlider = h("input", { type: "range", min: "0", max: "50", value: String(roiCut * 100) });
    const roiLabel = h("span", { class: "mono small" }, `${Math.round(roiCut * 100)}%`);

    function pushDebounced(name, value) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            const result = await api.post("/api/zed/param", { name, value });
            if (!result.ok) toast(`ros2 param set ${name} failed: ${result.output}`, "bad");
        }, 200);
    }

    autoToggle.addEventListener("change", () => {
        auto = autoToggle.checked;
        exposureSlider.disabled = auto;
        gainSlider.disabled = auto;
        pushDebounced("video.auto_exposure_gain", auto);
    });
    exposureSlider.addEventListener("input", () => {
        exposure = Number(exposureSlider.value);
        exposureLabel.textContent = String(exposure);
        pushDebounced("video.exposure", exposure);
    });
    gainSlider.addEventListener("input", () => {
        gain = Number(gainSlider.value);
        gainLabel.textContent = String(gain);
        pushDebounced("video.gain", gain);
    });
    roiSlider.addEventListener("input", () => {
        roiCut = Number(roiSlider.value) / 100;
        roiLabel.textContent = `${Math.round(roiCut * 100)}%`;
        roiOverlay.style.height = `${roiCut * 100}%`;
        pushDebounced("region_of_interest.manual_polygon", JSON.stringify(polygonForCut(roiCut)).replace(/\s/g, ""));
    });

    const roiOverlay = h("div", { class: "zed-roi-overlay", style: { height: `${roiCut * 100}%` } });
    const saveBtn = h("button", { class: "btn primary" }, icon("check"), "Save to repo");
    saveBtn.addEventListener("click", async () => {
        saveBtn.disabled = true;
        try {
            await api.post("/api/zed/save", {
                auto_exposure_gain: auto,
                exposure,
                gain,
                roi_polygon: polygonForCut(roiCut),
            });
            toast("Saved to jetson/cfr_arduino_bridge/config/cfr_zed2i.yaml", "good");
        } catch (e) {
            toast(e.message, "bad");
        } finally {
            saveBtn.disabled = false;
        }
    });

    root.replaceChildren(
        h(
            "div",
            { class: "page-head" },
            h(
                "div",
                {},
                h("h2", {}, "ZED exposure & ROI"),
                h("p", {}, "Live while connected to the car."),
            ),
        ),
        connectionCard,
        h(
            "div",
            { class: "grid cols-2" },
            card(
                "Live image",
                null,
                h("div", { class: "zed-stream-wrap" }, video, roiOverlay),
            ),
            card(
                "Tuning",
                null,
                h(
                    "div",
                    { class: "stack", style: { gap: "14px" } },
                    h("div", { class: "row" }, h("label", {}, "Auto exposure/gain"), autoToggle),
                    h("div", { class: "row" }, h("label", { style: { minWidth: "100px" } }, "Exposure"), exposureSlider, exposureLabel),
                    h("div", { class: "row" }, h("label", { style: { minWidth: "100px" } }, "Gain"), gainSlider, gainLabel),
                    h("div", { class: "row" }, h("label", { style: { minWidth: "100px" } }, "ROI top cut"), roiSlider, roiLabel),
                    h("div", { class: "note" }, "Sliders take effect immediately on the live node (ros2 param set) but are not persisted -- a relaunch reverts to whatever the file below says."),
                    saveBtn,
                ),
            ),
            card("ROI mask", "what the ROI cut actually removes (~/roi_mask/image)", h("div", { class: "zed-stream-wrap" }, roiMask)),
        ),
    );

    checkStatus();

    return () => {
        clearTimeout(debounceTimer);
        clearTimeout(statusTimer);
    };
}
