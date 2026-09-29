// Launch a plant-checkout profile on the car (straight_line_trim or
// figure_eight_calib), watch its log live, then review and apply what it
// measured -- the same analyze -> apply -> propagate/trim ->
// check_steering_consistency chain the manual workflow runs, one click at a
// time instead of four terminal commands.
//
// The hardware E-Stop interlock is unchanged: maneuver_runner_node.py still
// requires the witnessed assert/clear cycle before it arms anything. This
// page cannot skip that, and does not try to -- the confirmation below is a
// reminder, not a software gate on top of it.

import { api } from "../api.js";
import { h, icon, card, toast, empty } from "../ui.js";
import { jobs } from "../jobs.js";

export default async function calibrationPage(root) {
    let profiles = {};
    try {
        profiles = await api.get("/api/calibration/profiles");
    } catch (e) {
        root.replaceChildren(empty("Could not load profiles", e.message));
        return () => {};
    }

    const logCard = h("div");
    const resultCard = h("div");
    root.replaceChildren(
        h(
            "div",
            { class: "page-head" },
            h(
                "div",
                {},
                h("h2", {}, "Steering plant calibration"),
                h(
                    "p",
                    {},
                    "Runs the exact same characterize.launch.py profiles as the manual campaign (",
                    h("code", {}, "docs/characterization-steering.md"),
                    "), over ssh. The E-Stop remote is still the only thing that arms or aborts the car -- have it in hand before launching.",
                ),
            ),
        ),
        h(
            "div",
            { class: "stack" },
            h(
                "div",
                { class: "grid cols-2" },
                Object.entries(profiles).map(([name, info]) => profileCard(name, info)),
            ),
            logCard,
            resultCard,
        ),
    );

    function profileCard(name, info) {
        const launch = h("button", { class: "btn primary" }, icon("play"), "Launch");
        launch.addEventListener("click", () => {
            if (
                !confirm(
                    `Launch ${name} on the car?\n\nBe at the bench with the E-Stop remote in hand. This does not skip the hardware interlock -- the runner still waits for a witnessed E-Stop assert/clear cycle before it arms.`,
                )
            )
                return;
            startLaunch(name);
        });
        const running = jobs.active("calibrate", name);
        return card(
            name,
            info.propagation === "trim" ? "steering trim" : "broader plant re-fit",
            h(
                "div",
                { class: "stack", style: { gap: "10px" } },
                h("div", {}, info.description),
                running ? h("div", { class: "note" }, "Already running -- see the log below.") : launch,
            ),
        );
    }

    async function startLaunch(profile) {
        resultCard.replaceChildren();
        logCard.replaceChildren(card(`${profile} -- running`, null, h("div", { class: "muted" }, "Starting…")));
        await api.post("/api/calibration/launch", { profile });
        toast(`Launching ${profile}`);
        jobs.poll();
    }

    function renderLog(job) {
        if (!job) return;
        const lines = job.lines || [];
        logCard.replaceChildren(
            card(
                `${job.target} -- ${job.state}`,
                null,
                h(
                    "div",
                    { class: "stack", style: { gap: "8px" } },
                    job.state === "running" || job.state === "queued"
                        ? h("div", { class: "progress" }, h("div", { style: { width: `${job.progress * 100}%` } }))
                        : null,
                    h(
                        "pre",
                        { class: "mono small", style: { whiteSpace: "pre-wrap", maxHeight: "260px", overflow: "auto", margin: 0 } },
                        lines.join("\n"),
                    ),
                    job.state === "failed" ? h("div", { class: "note bad" }, job.error) : null,
                ),
            ),
        );
        if (job.state === "done" && job.result) renderResult(job.target, job.result);
    }

    function renderResult(profile, result) {
        const values = Object.entries(result.values || {});
        const applyBtn = h("button", { class: "btn primary" }, icon("check"), "Apply");
        applyBtn.addEventListener("click", async () => {
            applyBtn.disabled = true;
            try {
                const applied = await api.post(`/api/calibration/${result.run}/apply`, { profile });
                renderApplied(applied);
            } catch (e) {
                toast(e.message, "bad");
            } finally {
                applyBtn.disabled = false;
            }
        });
        resultCard.replaceChildren(
            card(
                `${result.run} -- measured values`,
                values.length ? `${values.length} value(s)` : "nothing measured",
                h(
                    "div",
                    { class: "stack", style: { gap: "10px" } },
                    values.length
                        ? h(
                              "table",
                              { class: "data" },
                              h("tbody", {}, values.map(([k, v]) => h("tr", {}, h("td", {}, h("code", {}, k)), h("td", {}, Array.isArray(v) ? JSON.stringify(v) : String(v))))),
                          )
                        : h("div", { class: "muted" }, "This run produced nothing to write into vehicle.yaml."),
                    h("div", { class: "row" }, values.length ? applyBtn : null, h("a", { class: "btn", href: `/api/runs/${result.run}/report.md` }, icon("download"), "Full report")),
                    h("div", { id: `apply-${result.run}` }),
                ),
            ),
        );
    }

    function renderApplied(applied) {
        const holder = document.getElementById(`apply-${applied.run}`);
        if (!holder) return;
        holder.replaceChildren(
            h(
                "div",
                { class: "stack", style: { gap: "8px" } },
                applied.steps.map(([name, step]) =>
                    h(
                        "div",
                        {},
                        h("div", { class: `row` }, step.ok ? icon("check") : icon("x"), h("b", {}, name)),
                        h("pre", { class: "mono small", style: { whiteSpace: "pre-wrap", maxHeight: "160px", overflow: "auto", margin: 0 } }, step.output),
                    ),
                ),
                applied.consistent
                    ? h("div", { class: "note good" }, "check_steering_consistency.py passed -- every copy agrees.")
                    : h("div", { class: "note bad" }, "check_steering_consistency.py found a disagreement -- see its output above before trusting this."),
            ),
        );
    }

    const off = jobs.onChange((changed) => {
        for (const job of changed) {
            if (job.kind === "calibrate") renderLog(job);
        }
    });
    // A job already running when this page opens (e.g. after a reload).
    const existing = jobs.list.find((j) => j.kind === "calibrate" && (j.state === "running" || j.state === "queued"));
    if (existing) renderLog(existing);

    return off;
}
