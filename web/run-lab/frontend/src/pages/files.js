// Everything in the run directory, grouped, each downloadable -- plus the
// whole run as one archive, and the commands to open the bag elsewhere.

import { api } from "../api.js";
import { h, icon, card, bytes, table, toast } from "../ui.js";

export default async function filesPage(root, { run, data }) {
    const files = await data.files();
    const groups = {};
    for (const f of files) (groups[f.group] ||= []).push(f);
    const total = files.reduce((a, f) => a + f.bytes, 0);
    const health = await api.get("/api/health");
    const runPath = `${health.runs_dir}/${run}`;

    const clearBtn = h("button", { class: "btn danger" }, icon("trash"), "Delete analysis");
    clearBtn.addEventListener("click", async () => {
        if (!confirm("Delete the processed analysis for this run?  The bag and logs are kept; you can re-process at any time.")) return;
        await api.del(`/api/runs/${encodeURIComponent(run)}?analysis_only=true`);
        toast("Analysis deleted");
        location.hash = `#/run/${encodeURIComponent(run)}/overview`;
        location.reload();
    });

    const cmds = h(
        "pre",
        { class: "mono small", style: { whiteSpace: "pre-wrap", margin: 0 } },
        `# the analysed run in the native Rerun viewer (course, car, clouds, camera, channels, logs)\nweb/run-lab/.venv/bin/rerun ${runPath}/analysis/recording.rrd\n\n# the raw bag, every topic as recorded (Rerun reads MCAP directly)\nweb/run-lab/.venv/bin/rerun ${runPath}/bag/*.mcap\n\n# inspect, or play into a live ROS graph (RViz, your own nodes)\nros2 bag info ${runPath}/bag\nros2 bag play ${runPath}/bag --clock\n\n# re-run the analysis from a terminal\nweb/run-lab/.venv/bin/python web/run-lab/server/analyze.py ${runPath}`,
    );

    root.append(
        h(
            "div",
            { class: "page-head" },
            h("div", {}, h("h2", {}, "Files"), h("p", {}, `${files.length} files · ${bytes(total)}.  The bag is MCAP, which Rerun opens as-is; analysis/recording.rrd is the analysed run for Rerun.`)),
            h("span", { class: "spacer" }),
            h("a", { class: "btn primary", href: `/api/runs/${run}/archive` }, icon("download"), "Download run (.tar)"),
            h("a", { class: "btn", href: `/api/runs/${run}/archive?analysis=true` }, "…with analysis"),
            clearBtn,
        ),
        h(
            "div",
            { class: "stack" },
            card("Open elsewhere", runPath, cmds),
            ...Object.entries(groups).map(([g, fs]) =>
                card(
                    g,
                    `${fs.length} files · ${bytes(fs.reduce((a, f) => a + f.bytes, 0))}`,
                    table(
                        [
                            { label: "Path", get: (f) => h("span", { class: "mono" }, f.path) },
                            { label: "Size", num: true, get: (f) => bytes(f.bytes) },
                            { label: "", get: (f) => h("a", { class: "btn sm", href: `/api/runs/${run}/file?path=${encodeURIComponent(f.path)}` }, icon("download")) },
                        ],
                        fs,
                    ),
                ),
            ),
        ),
    );
}
