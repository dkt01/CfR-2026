// Thin fetch layer, plus a per-run cache so switching tabs does not refetch
// a half-megabyte series.

async function request(method, path, body) {
    const res = await fetch(path, {
        method,
        // Always ask the server.  Re-processing rewrites a run's files under
        // the same URLs, and a copy the browser cached before the server sent
        // Cache-Control could otherwise be served without asking; an
        // unchanged file costs a 304.
        cache: "no-cache",
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try {
            const j = await res.json();
            detail = j.detail || j.error || detail;
        } catch {}
        throw new Error(detail);
    }
    const type = res.headers.get("content-type") || "";
    if (type.includes("json")) return res.json();
    if (type.includes("octet-stream")) return res.arrayBuffer();
    return res.text();
}

export const api = {
    get: (p) => request("GET", p),
    post: (p, b = {}) => request("POST", p, b),
    del: (p) => request("DELETE", p),
};

const cache = new Map();

export function runData(name) {
    if (!cache.has(name)) cache.set(name, {});
    const c = cache.get(name);
    const once = (key, path) => (c[key] ??= api.get(path).catch((e) => { delete c[key]; throw e; }));
    return {
        summary: () => once("summary", `/api/runs/${name}/summary`),
        series: () => once("series", `/api/runs/${name}/series`).then(unpackSeries),
        course: () => once("course", `/api/runs/${name}/course`),
        files: () => api.get(`/api/runs/${name}/files`),
    };
}

export function dropRun(name) {
    cache.delete(name);
}

// series.json carries nulls for gaps; typed arrays with NaN are what the
// charts and the map want.
const unpacked = new WeakMap();
function unpackSeries(raw) {
    if (unpacked.has(raw)) return unpacked.get(raw);
    const cols = {};
    for (const [k, arr] of Object.entries(raw.columns)) {
        const a = new Float64Array(arr.length);
        for (let i = 0; i < arr.length; i++) a[i] = arr[i] === null ? NaN : arr[i];
        cols[k] = a;
    }
    const out = { hz: raw.hz, run_t0: raw.run_t0, run_t1: raw.run_t1, cols, n: cols.t.length };
    unpacked.set(raw, out);
    return out;
}

// Index of the sample at or before time t.
export function indexAt(series, t) {
    const ts = series.cols.t;
    if (!ts.length) return 0;
    const i = Math.round((t - ts[0]) * series.hz);
    return Math.max(0, Math.min(ts.length - 1, i));
}
