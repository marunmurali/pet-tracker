"use strict";

const $ = (id) => document.getElementById(id);
const pad = (n) => String(n).padStart(2, "0");
const hhmm = (ts) => { const d = new Date(ts * 1000); return `${pad(d.getHours())}:${pad(d.getMinutes())}`; };
const dur = (s) => { const m = Math.round(s / 60); return m >= 60 ? `${Math.floor(m / 60)}h ${pad(m % 60)}m` : `${m}m`; };
const localDate = (d = new Date()) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  if (r.status === 401) { location.href = "/"; throw new Error("unauthorized"); }
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function renderState(s) {
  const conn = $("conn");
  if (!s.enabled) { conn.textContent = "Off"; conn.className = "pill off"; }
  else if (s.connected) { conn.textContent = "Live"; conn.className = "pill ok"; }
  else { conn.textContent = "Connecting"; conn.className = "pill warn"; }

  $("state").textContent = !s.enabled ? "Off" : (s.state || "–") + (s.zone ? ` · ${s.zone}` : "");
  $("since").textContent = s.enabled && s.since ? `since ${hhmm(s.since)} (${dur(s.now - s.since)})` : "";
  $("barks-today").textContent = s.barks_today;
  $("enabled").checked = s.enabled;
  $("tracking").checked = s.tracking;
  $("tracking").disabled = s.ptz === "none";

  $("follow").textContent = !s.enabled ? ""
    : s.sweeping ? `🔍 Searching for ${s.dog_name}…`
    : s.locked ? `🔒 Following ${s.dog_name}${s.lock_source ? ` (${s.lock_source})` : ""}` : "";
  $("training").textContent = s.training_images != null ? `${s.training_images} training pictures` : "";
}

async function refreshState() {
  try { renderState(await api("/api/state")); } catch (e) { console.warn(e); }
}

async function toggle(path, key, el) {
  el.disabled = true;
  try {
    renderState(await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [key]: el.checked }),
    }));
    // restart the MJPEG connection so it reflects the new on/off state immediately
    if (key === "enabled") $("live").src = `/stream.mjpg?t=${Date.now()}`;
  } catch (e) {
    el.checked = !el.checked;
  } finally {
    el.disabled = false;
    refreshState();
  }
}

async function refreshDay() {
  const day = $("day").value || localDate();
  const [tl, barks] = await Promise.all([
    api(`/api/timeline?date=${day}`),
    api(`/api/barks?date=${day}`),
  ]);
  const span = tl.day_end - tl.day_start;
  const pct = (ts) => `${(100 * (ts - tl.day_start) / span).toFixed(3)}%`;

  const box = $("timeline");
  box.replaceChildren();
  for (const iv of tl.intervals) {
    const seg = document.createElement("div");
    seg.className = `seg ${iv.state}`;
    seg.style.left = pct(iv.start);
    seg.style.width = `${(100 * (iv.end - iv.start) / span).toFixed(3)}%`;
    seg.title = `${iv.state} ${hhmm(iv.start)}–${hhmm(iv.end)}`;
    box.appendChild(seg);
  }
  for (const b of barks) {
    const tick = document.createElement("div");
    tick.className = "bark";
    tick.style.left = pct(b.ts);
    box.appendChild(tick);
  }
  const now = Date.now() / 1000;
  if (now > tl.day_start && now < tl.day_end) {
    const line = document.createElement("div");
    line.className = "now";
    line.style.left = pct(now);
    box.appendChild(line);
  }
  for (const k of ["resting", "active", "absent"]) $(`t-${k}`).textContent = dur(tl.per_state[k] || 0);

  const list = $("bark-list");
  list.replaceChildren();
  if (!barks.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No barks.";
    list.appendChild(li);
  }
  for (const b of barks) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${hhmm(b.ts)}  ·  score ${b.data.score ?? "?"}${b.data.alerted ? "" : " (no alert)"}`;
    li.appendChild(label);
    if (b.clip_url) {
      const btn = document.createElement("button");
      btn.textContent = "Play";
      btn.onclick = () => {
        const p = $("player");
        p.src = b.clip_url;
        p.hidden = false;
        p.play().catch(() => {});
        p.scrollIntoView({ behavior: "smooth", block: "nearest" });
      };
      li.appendChild(btn);
    }
    list.appendChild(li);
  }
}

$("enabled").addEventListener("change", (e) => toggle("/api/enabled", "enabled", e.target));
$("tracking").addEventListener("change", (e) => toggle("/api/tracking", "tracking", e.target));
$("day").value = localDate();
$("day").addEventListener("change", () => refreshDay().catch(console.warn));

refreshState();
refreshDay().catch(console.warn);
setInterval(refreshState, 3000);
setInterval(() => { if ($("day").value === localDate()) refreshDay().catch(console.warn); }, 60000);
// Safari stops MJPEG when the tab is backgrounded; reconnect when it comes back
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) $("live").src = `/stream.mjpg?t=${Date.now()}`;
});
