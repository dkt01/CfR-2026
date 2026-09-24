// Small DOM + formatting helpers.  No framework: pages render into a node
// and wire their own listeners; the playhead store is the only shared state.

export function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
        else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
        else if (k === "html") el.innerHTML = v;
        else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat(Infinity)) {
        if (c === null || c === undefined || c === false) continue;
        el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);

export function fmt(v, digits = 2, unit = "") {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (typeof v === "string") return v;
    const s = Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toFixed(digits);
    return unit ? `${s} ${unit}` : s;
}

export function bytes(n) {
    if (n === null || n === undefined) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (n >= 1000 && i < u.length - 1) {
        n /= 1000;
        i++;
    }
    return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

export function clock(t) {
    if (t === null || t === undefined || Number.isNaN(t)) return "—";
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

export function when(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// Icons: a handful of inline strokes, sized by CSS.
const PATHS = {
    car: "M5 17h14M6 17l1.5-5h9L18 17M7.5 12l1.2-3.5h6.6l1.2 3.5M7 17v2m10-2v2",
    runs: "M4 6h16M4 12h16M4 18h10",
    overview: "M4 13h6V4H4zm10 7h6V11h-6zM4 20h6v-4H4zm10-11h6V4h-6z",
    track: "M6 18c-2 0-3-1.5-3-3.5S5 11 7 11h10c2 0 4-1.5 4-3.5S19 4 17 4H9",
    chart: "M4 19V5m0 14h16M8 15l3-4 3 2 5-6",
    wheel: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zm0 6a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM3.5 10.5h5.5m6 0h5.5M12 15v6",
    brain: "M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V5a2 2 0 0 0-3-1zm6 0a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1",
    cube: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
    cpu: "M7 7h10v10H7zM9 3v4m6-4v4M9 17v4m6-4v4M3 9h4m-4 6h4m10-6h4m-4 6h4",
    logs: "M5 5h14M5 9h14M5 13h9M5 17h6",
    files: "M6 3h8l4 4v14H6zM14 3v4h4",
    play: "M8 5v14l11-7z",
    pause: "M7 5h4v14H7zm6 0h4v14h-4z",
    replay: "M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4",
    download: "M12 4v11m0 0l-4-4m4 4l4-4M5 20h14",
    refresh: "M20 12a8 8 0 1 1-2.3-5.6M20 4v4h-4",
    trash: "M5 7h14M10 7V5h4v2M7 7l1 13h8l1-13",
    check: "M5 12.5l4.5 4.5L19 7.5",
    x: "M6 6l12 12M18 6L6 18",
    warn: "M12 4l9 16H3zM12 10v4m0 3v.5",
    info: "M12 8v.5M12 11v6M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z",
    link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    usb: "M12 3v14m0 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4zM12 7l-4 4v2m4-2l4-3V6",
    gazebo: "M4 20l8-16 8 16zM8 13h8",
    external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
};

export function icon(name, cls = "") {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "1.8");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    if (cls) svg.setAttribute("class", cls);
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", PATHS[name] || PATHS.info);
    svg.append(p);
    return svg;
}

// Status is never colour alone: an icon and a word ride with it.
export function statusBadge(severity, text) {
    const map = { good: ["good", "check"], warn: ["warn", "warn"], bad: ["bad", "x"], info: ["info", "info"] };
    const [cls, ic] = map[severity] || map.info;
    return h("span", { class: `badge ${cls}` }, icon(ic, "status-icon"), text);
}

export function card(title, sub, body, tools) {
    return h(
        "section",
        { class: "card" },
        title ? h("div", { class: "card-head" }, h("h3", {}, title), sub ? h("span", { class: "sub" }, sub) : null, h("span", { class: "spacer" }), tools || null) : null,
        h("div", { class: "card-body" }, body),
    );
}

export function tile(label, value, unit, tone, hint) {
    return h(
        "div",
        { class: `tile ${tone ? `tone-${tone}` : ""}` },
        h("div", { class: "label" }, label),
        h("div", { class: "value" }, value === null || value === undefined ? "—" : value, unit ? h("small", {}, unit) : null),
        hint ? h("div", { class: "hint" }, hint) : null,
    );
}

export function kv(pairs) {
    const dl = h("dl", { class: "kv" });
    for (const [k, v] of pairs) {
        if (v === undefined) continue;
        dl.append(h("dt", {}, k), h("dd", {}, v === null ? "—" : v));
    }
    return dl;
}

export function table(columns, rows, onRow) {
    const t = h("table", { class: "data" });
    t.append(h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "" }, c.label)))));
    const tb = h("tbody");
    for (const r of rows) {
        const tr = h("tr", { class: onRow ? "clickable" : "" }, columns.map((c) => {
            const v = c.get ? c.get(r) : r[c.key];
            return h("td", { class: c.num ? "num" : "" }, v instanceof Node ? v : v === null || v === undefined ? "—" : v);
        }));
        if (onRow) tr.addEventListener("click", () => onRow(r));
        tb.append(tr);
    }
    t.append(tb);
    return h("div", { class: "table-wrap" }, t);
}

export function toast(text, tone = "") {
    const el = h("div", { class: `toast ${tone}` }, text);
    document.getElementById("toasts").append(el);
    setTimeout(() => el.remove(), 5200);
}

export function empty(title, text, action) {
    return h("div", { class: "empty" }, h("h3", {}, title), h("div", {}, text), action ? h("div", { style: { marginTop: "14px" } }, action) : null);
}

export function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function segmented(options, value, onChange) {
    const wrap = h("div", { class: "seg" });
    for (const [key, label] of options) {
        const b = h("button", { class: key === value ? "on" : "" }, label);
        b.addEventListener("click", () => {
            wrap.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
            b.classList.add("on");
            onChange(key);
        });
        wrap.append(b);
    }
    return wrap;
}
