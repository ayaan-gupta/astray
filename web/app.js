/* Astray frontend. Three routes, one flow, no build step.
 *
 * The load-bearing behaviour here is the SSE handling on the session route.
 * A run takes ~3-4 minutes, and the diagnosis lands in the first 15-30s, so the
 * page renders the diagnosis card and enables chat the moment `diagnosis_ready`
 * arrives rather than waiting for the video. Everything after that fills in:
 * beats appear greyed as s6 plans them, then become seekable when the render
 * reports measured timings. `done` means the whole run finished, not just the
 * diagnosis -- see the /stream handler.
 *
 * Beat citations are the reason chat is worth having. The server validates
 * `[beat:bN]` against the manifest and strips unknown ids, so anything that
 * reaches here names a real beat; we turn it into a button that seeks the
 * player to that beat's measured start.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `request failed (${r.status})`);
  return body;
});

// Anonymous, browser-local. Enables per-student history with no accounts.
const handle = (() => {
  let h = localStorage.getItem("astray.handle");
  if (!h) { h = "s-" + Math.random().toString(36).slice(2, 10); localStorage.setItem("astray.handle", h); }
  return h;
})();

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const mount = (id) => {
  const view = $("#view");
  view.replaceChildren($(id).content.cloneNode(true));
  return view;
};

/* ------------------------------------------------------------------ routing */
function router() {
  const hash = location.hash || "#/";
  document.querySelectorAll("nav a").forEach((a) =>
    a.classList.toggle("is-active", a.getAttribute("href") === hash.split("/").slice(0, 2).join("/")));
  if (hash.startsWith("#/session/")) return renderSession(hash.split("/")[2]);
  if (hash === "#/insights") return renderInsights();
  return renderSubmit();
}
window.addEventListener("hashchange", router);
window.addEventListener("DOMContentLoaded", router);

/* ------------------------------------------------------------------- submit */
function renderSubmit() {
  mount("#tpl-submit");
  const err = $("#submit-err");
  const show = (name) => {
    $("#form-typed").classList.toggle("hidden", name !== "typed");
    $("#form-photo").classList.toggle("hidden", name !== "photo");
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("is-active", t.dataset.tab === name));
  };
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => show(t.dataset.tab)));

  $("#form-typed").onsubmit = async (e) => {
    e.preventDefault();
    err.textContent = "";
    const f = new FormData(e.target);
    try {
      const { session_id } = await api("/api/sessions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ handle, problem: f.get("problem"), work: f.get("work") }),
      });
      location.hash = `#/session/${session_id}`;
    } catch (e2) { err.textContent = e2.message; }
  };

  /* Photo path: upload, then REVIEW before anything is diagnosed. The server
   * refuses to diagnose an unconfirmed transcription, so this step is not
   * cosmetic -- a transcription the student never checked could pin a
   * misconception on them for an error the vision model invented. */
  const drop = $("#drop");
  const fileInput = $("input[type=file]", drop);
  drop.onclick = () => fileInput.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("is-over"); };
  drop.ondragleave = () => drop.classList.remove("is-over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("is-over"); fileInput.files = e.dataTransfer.files; fileInput.onchange(); };

  let sessionId = null;
  fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;
    err.textContent = "";
    const problem = $("#form-photo input[name=problem]").value || "(from photo)";
    try {
      const created = await api("/api/sessions", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ handle, problem, work: "" }),
      });
      sessionId = created.session_id;
      $("span", drop).textContent = "Reading your handwriting…";
      const fd = new FormData();
      fd.append("file", file);
      const res = await api(`/api/sessions/${sessionId}/photo`, { method: "POST", body: fd });
      $("span", drop).textContent = file.name;
      showReview(res.transcription);
    } catch (e2) { err.textContent = e2.message; $("span", drop).textContent = "Try another photo"; }
  };

  function showReview(t) {
    $("#review").classList.remove("hidden");
    const box = $("#review-lines");
    box.replaceChildren();
    const mk = (value, cls) => {
      const i = document.createElement("input");
      i.value = value; i.className = cls; box.appendChild(i); return i;
    };
    const problemInput = mk(t.problem, "review-problem");
    const stepInputs = (t.steps.length ? t.steps : [""]).map((s) => mk(s, "review-step"));
    $("#confirm-btn").onclick = async () => {
      try {
        await api(`/api/sessions/${sessionId}/submission`, {
          method: "PUT", headers: { "content-type": "application/json" },
          body: JSON.stringify({
            problem: problemInput.value,
            work: stepInputs.map((i) => i.value).filter((v) => v.trim()).join("\n"),
          }),
        });
        location.hash = `#/session/${sessionId}`;
      } catch (e2) { err.textContent = e2.message; }
    };
  }
}

/* ------------------------------------------------------------------ session */
const STAGES = [
  ["s1_diagnose", "Diagnosing your working"],
  ["s2_intent", "Working out what you need"],
  ["s3_prereq", "Mapping prerequisites"],
  ["s4_curriculum", "Sequencing the explanation"],
  ["s5_math", "Choosing the maths to show"],
  ["s6_visual", "Storyboarding the animation"],
  ["s7_scene", "Writing the animation"],
  ["s8_validate", "Checking it is safe to run"],
];

function renderSession(sessionId) {
  mount("#tpl-session");
  const stageList = $("#stage-list");
  const done = new Set();
  const drawStages = (active) => {
    stageList.replaceChildren(...STAGES.map(([id, label]) => {
      const li = document.createElement("li");
      li.className = id === active ? "active" : "";
      li.innerHTML = `<span class="tick">${done.has(id) ? "✓" : id === active ? "▸" : "·"}</span><span></span>`;
      li.lastChild.textContent = label;
      return li;
    }));
  };
  drawStages("s1_diagnose");

  let beats = [];
  let video = null;

  const drawRail = () => {
    const rail = $("#rail");
    rail.replaceChildren(...beats.map((b) => {
      const el = document.createElement("button");
      el.className = "beat" + (b.targets_misconception ? " is-target" : "");
      el.disabled = b.start_s === null || !video;
      el.innerHTML = `<span></span><span class="t"></span>`;
      el.firstChild.textContent = b.title;
      el.lastChild.textContent = b.start_s === null ? "planned" : fmt(b.start_s);
      el.onclick = () => seek(b.id);
      rail.appendChild(el);
      return el;
    }));
  };

  const seek = (beatId) => {
    const b = beats.find((x) => x.id === beatId);
    if (!b || b.start_s === null || !video) return;
    video.currentTime = b.start_s + 0.05;
    video.play();
    document.querySelectorAll(".beat").forEach((el, i) =>
      el.classList.toggle("is-active", beats[i].id === beatId));
  };

  const showVideo = (url) => {
    const wrap = $("#player-wrap");
    wrap.classList.remove("empty");
    video = document.createElement("video");
    video.src = url; video.controls = true; video.playsInline = true;
    wrap.replaceChildren(video);
    video.ontimeupdate = () => {
      const t = video.currentTime;
      document.querySelectorAll(".beat").forEach((el, i) => {
        const b = beats[i];
        el.classList.toggle("is-active", b.start_s !== null && t >= b.start_s && t < b.end_s);
      });
    };
    drawRail();
  };

  /* Chat opens on `diagnosis_ready`, not on the video. Waiting for the render
   * would leave the student with nothing to do for three minutes. */
  const enableChat = () => {
    $("#chat-input").disabled = false;
    $("#chat-form button").disabled = false;
    $("#chat-hint").textContent = "Ask anything — answers cite moments in your animation.";
  };

  const addMessage = (role, text, cited) => {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    const who = document.createElement("div");
    who.className = "who";
    who.textContent = role === "user" ? "You" : "Tutor";
    const body = document.createElement("div");
    body.className = "body";
    // Citations arrive server-validated; render them as seek buttons.
    const parts = text.split(/(\[beat:b\d+\])/g);
    parts.forEach((p) => {
      const m = p.match(/^\[beat:(b\d+)\]$/);
      if (!m) return body.appendChild(document.createTextNode(p));
      const b = beats.find((x) => x.id === m[1]);
      const btn = document.createElement("button");
      btn.className = "cite";
      btn.textContent = b ? `▶ ${b.start_s === null ? b.title : fmt(b.start_s) + " — " + b.title}` : m[1];
      btn.onclick = () => seek(m[1]);
      body.appendChild(btn);
    });
    el.append(who, body);
    $("#messages").append(el);
    $("#messages").scrollTop = $("#messages").scrollHeight;
  };

  $("#chat-form").onsubmit = async (e) => {
    e.preventDefault();
    const input = $("#chat-input");
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    addMessage("user", q, []);
    try {
      const res = await api(`/api/sessions/${sessionId}/chat`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ message: q }),
      });
      addMessage("assistant", res.reply, res.cited_beats);
    } catch (e2) { addMessage("assistant", `(${e2.message})`, []); }
  };

  const showDiagnosis = async (d) => {
    const card = $("#diagnosis");
    card.classList.remove("pending");
    if (d.no_error_found) {
      card.innerHTML = `<h1>Your working is correct</h1>`;
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = d.misconception_statement;
      card.append(p);
      return;
    }
    card.replaceChildren();
    const h = document.createElement("h1");
    h.textContent = "Here's where it went astray";
    const rule = document.createElement("div");
    rule.className = "rule";
    rule.textContent = d.buggy_rule;
    const stmt = document.createElement("p");
    stmt.textContent = d.misconception_statement;
    const badges = document.createElement("div");
    badges.className = "badges";
    const badge = (text, cls = "") => {
      const b = document.createElement("span");
      b.className = "badge " + cls; b.textContent = text; badges.append(b);
    };
    if (d.verified_by_sympy) badge("✓ checked with SymPy", "ok");
    else badge("not symbolically checkable");
    if (d.divergence_index !== null && d.divergence_index !== undefined)
      badge(`diverges at step ${d.divergence_index + 1}`);
    badge(`confidence ${Math.round(d.confidence * 100)}%`);
    card.append(h, rule, stmt, badges);

    try {
      const peers = await api(`/api/sessions/${sessionId}/peers`);
      if (peers.others > 0) badge(`${peers.others} other student${peers.others > 1 ? "s" : ""} made this error`, "peers");
    } catch { /* insights are additive; never block the diagnosis on them */ }
  };

  /* Render whatever is already persisted before opening the stream.
   *
   * This is not just an optimisation. Reconnecting to a finished session replays
   * only `diagnosis_ready` and `done` -- by design, so a reconnect does not
   * re-run and re-bill the pipeline -- which means `render_complete` never
   * fires again. A page that only learned about beats, the video and chat
   * history from live events would come back from a reload with an empty
   * theatre and an empty conversation. Beats load before chat so citations in
   * the history can resolve to real timestamps rather than bare ids. */
  (async () => {
    const [session, info] = await Promise.all([
      api(`/api/sessions/${sessionId}`),
      api(`/api/sessions/${sessionId}/beats`).catch(() => ({ beats: [], video_url: null })),
    ]);
    if (session.diagnosis) { showDiagnosis(session.diagnosis); enableChat(); }
    beats = info.beats;
    drawRail();
    if (info.video_url) showVideo(info.video_url);
    const history = await api(`/api/sessions/${sessionId}/chat`).catch(() => ({ messages: [] }));
    history.messages.forEach((m) => addMessage(m.role, m.content, m.cited_beats));
  })().catch((e) => console.error("session init failed", e));

  const es = new EventSource(`/api/sessions/${sessionId}/stream`);
  es.addEventListener("stage_started", (e) => drawStages(JSON.parse(e.data).stage));
  es.addEventListener("stage_completed", (e) => {
    const ev = JSON.parse(e.data);
    done.add(ev.stage);
    const next = STAGES[STAGES.findIndex(([id]) => id === ev.stage) + 1];
    drawStages(next ? next[0] : null);
    if (ev.stage === "s6_visual" && ev.payload?.beats) {
      beats = ev.payload.beats.map((b) => ({ ...b, start_s: null, end_s: null, targets_misconception: false }));
      drawRail();
    }
  });
  es.addEventListener("diagnosis_ready", (e) => { showDiagnosis(JSON.parse(e.data).payload); enableChat(); });
  es.addEventListener("render_complete", async () => {
    const info = await api(`/api/sessions/${sessionId}/beats`);
    beats = info.beats;
    if (info.video_url) showVideo(info.video_url);
    drawRail();
  });
  es.addEventListener("error", (e) => {
    try { $("#pipeline").textContent = JSON.parse(e.data).message || "something went wrong"; } catch { /* connection blip */ }
  });
  es.addEventListener("done", () => es.close());
}

/* ----------------------------------------------------------------- insights */
async function renderInsights() {
  mount("#tpl-insights");
  const data = await api(`/api/insights?handle=${encodeURIComponent(handle)}`);
  const rows = (items, empty, fmtRow) => {
    if (!items.length) return `<p class="muted">${empty}</p>`;
    return items.map(fmtRow).join("");
  };
  $("#history").innerHTML = rows(data.history, "Nothing yet — submit some work and it will show up here.",
    (m) => `<div class="row"><span>${esc(m.canonical_statement)}</span><span class="n">${m.times}×</span></div>`);
  $("#frequency").innerHTML = rows(data.misconceptions, "No diagnoses logged yet.",
    (m) => `<div class="row"><span>${esc(m.canonical_statement)}</span><span class="n">${m.students} student${m.students > 1 ? "s" : ""}</span></div>`);
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
