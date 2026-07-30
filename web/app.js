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
 *
 * Model text is never assigned to innerHTML. It is tokenised and rendered into
 * elements built here, so the markdown the tutor writes becomes real formatting
 * without the reply ever being parsed as markup.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
/* "Here's where it went <Astray>", with the last word set as the wordmark.
 *
 * The logo is a masked box rather than an <img>: the lettering is a single path
 * filled `currentColor`, which an <img> renders in its own document context
 * where `currentColor` is black -- invisible on this page. Painting
 * `background: currentColor` through the artwork as a mask inverts that, so the
 * word takes the exact colour of the headline it sits in and follows it through
 * any future change of colour or theme.
 *
 * The word carries an accessible name, so the heading still reads as a whole
 * sentence: "Here's where it went Astray".
 */
const headline = () => {
  const h = el("h1", null, "Here's where it went ");
  const word = el("span", "wordmark");
  word.setAttribute("role", "img");
  word.setAttribute("aria-label", "Astray");
  h.append(word);
  return h;
};

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

/* Cursor spotlight. The gradient itself lives in CSS on `.card-lit::before`
 * and reads --mx/--my; all this does is keep those two custom properties in
 * step with the pointer. Bound per card on pointerenter rather than globally on
 * the document, so a page of cards costs one listener on the one being hovered.
 * A device with no pointer never fires this and the vars stay unset, which is
 * exactly the state where the layer renders nothing. */
const lightCards = (root) => {
  root.querySelectorAll(".card-lit").forEach((card) => {
    card.addEventListener("pointermove", (e) => {
      const box = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - box.left}px`);
      card.style.setProperty("--my", `${e.clientY - box.top}px`);
    });
  });
};

/* The entrance animation ends on `filter: blur(0)`, and with a forwards fill
 * that declaration persists. A lingering filter -- even a zero-radius one --
 * keeps the element on its own compositing layer and drops text from subpixel
 * to greyscale antialiasing, which is a visible quality loss on a dark
 * background. Dropping the class once the animation finishes removes it. */
const cleanUpEntrance = (root) => {
  root.querySelectorAll(".enter").forEach((node) =>
    node.addEventListener("animationend", () => node.classList.remove("enter"), { once: true }));
};

const mount = (id) => {
  const view = $("#view");
  view.replaceChildren($(id).content.cloneNode(true));
  lightCards(view);
  cleanUpEntrance(view);
  return view;
};

/* ------------------------------------------------------------------ routing */
function router() {
  const hash = location.hash || "#/";
  document.querySelectorAll(".nav a").forEach((a) =>
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
  const tabs = [...document.querySelectorAll(".tab")];

  const show = (name) => {
    $("#form-typed").classList.toggle("hidden", name !== "typed");
    $("#form-photo").classList.toggle("hidden", name !== "photo");
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
    });
  };
  tabs.forEach((t, i) => {
    t.onclick = () => show(t.dataset.tab);
    // APG tab pattern: arrows move between tabs, Tab leaves the tablist.
    t.onkeydown = (e) => {
      const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      e.preventDefault();
      const next = tabs[(i + step + tabs.length) % tabs.length];
      show(next.dataset.tab);
      next.focus();
    };
  });

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
  const fileInput = $("#photo-file");
  const dropText = $("#drop-text");
  // The <label for> already opens the picker on click and on Enter from the
  // focused input; adding a click handler here would open it twice.
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("is-over"); };
  drop.ondragleave = () => drop.classList.remove("is-over");
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove("is-over");
    fileInput.files = e.dataTransfer.files;
    fileInput.onchange();
  };

  let sessionId = null;
  fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;
    err.textContent = "";
    const problem = $("#photo-problem").value || "(from photo)";
    try {
      const created = await api("/api/sessions", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ handle, problem, work: "" }),
      });
      sessionId = created.session_id;
      dropText.textContent = "Reading your handwriting…";
      const fd = new FormData();
      fd.append("file", file);
      const res = await api(`/api/sessions/${sessionId}/photo`, { method: "POST", body: fd });
      dropText.textContent = file.name;
      showReview(res.transcription);
    } catch (e2) {
      err.textContent = e2.message;
      dropText.textContent = "Choose another photo";
    }
  };

  function showReview(t) {
    const review = $("#review");
    review.classList.remove("hidden");
    const box = $("#review-lines");
    box.replaceChildren();
    const mk = (value, cls, label) => {
      const i = el("input", cls);
      i.value = value;
      i.setAttribute("aria-label", label);
      box.appendChild(i);
      return i;
    };
    const problemInput = mk(t.problem, "review-problem", "Problem");
    const steps = t.steps.length ? t.steps : [""];
    const stepInputs = steps.map((s, i) => mk(s, "review-step", `Step ${i + 1}`));
    review.scrollIntoView({ block: "nearest", behavior: "smooth" });
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

  showPastProblems();
}

/* The student's own shelf of past work.
 *
 * Additive, like insights: a failure here leaves the card hidden rather than
 * blocking the form that is the point of this page. The card starts `hidden` in
 * the template and is only revealed once there is something in it, so a
 * first-time visitor never sees an empty shelf asking to be filled.
 *
 * Voice is the reason this is on the front page rather than tucked into
 * insights. A problem created by speaking has no form submission behind it and
 * no URL the student typed, so without a list there is no route back to it. */
async function showPastProblems() {
  const card = $("#past");
  if (!card) return;
  let sessions = [];
  try {
    ({ sessions } = await api(`/api/sessions?handle=${encodeURIComponent(handle)}`));
  } catch { return; }
  if (!sessions.length) return;

  $("#past-list").replaceChildren(...sessions.map((s) => {
    const row = el("a", "past-row");
    row.href = `#/session/${s.session_id}`;

    const main = el("span", "past-main");
    main.append(el("span", "past-problem mono", s.problem));
    // What it turned out to be, which is the reason to click one row over
    // another. A session still running has no diagnosis yet, and one whose
    // working was correct never gets a misconception, so both fall back to a
    // state -- shown as a state, not as an error.
    main.append(el("span", "past-sub", subtitleFor(s)));
    row.append(main);

    if (s.topic) row.append(el("span", "past-topic", prettyTopic(s.topic)));
    row.append(el("time", "past-when", relativeDay(s.created_at)));
    return row;
  }));
  card.hidden = false;
}

/* The taxonomy's topics are machine keys: `algebra.binomial_expansion`. They are
 * built to be counted and joined on, so they are shown here as the last segment
 * in words rather than renamed at the source, where a display label would have to
 * stay stable across every row that already references the key. */
const prettyTopic = (topic) => {
  const leaf = String(topic).split(".").pop().replace(/[_-]+/g, " ").trim();
  return leaf ? leaf[0].toUpperCase() + leaf.slice(1) : "";
};

/* What a row says when there is no misconception to name.
 *
 * `ready` is the ambiguous one, and the reason this is a function rather than a
 * lookup: a finished explainer and finished correct working both end at `ready`
 * with no misconception, and telling a student who never showed their working
 * that their working was correct is a different claim entirely. */
const subtitleFor = (s) => {
  if (s.misconception) return s.misconception;
  if (s.explaining) return PAST_STATUS.explaining;
  return PAST_STATUS[s.status] || s.status;
};

const PAST_STATUS = {
  created: "Not started yet",
  in_progress: "Still working on this one",
  ready: "Correct working, nothing to fix",
  explaining: "Explained, no working to correct",
  needs_clarification: "Needs your working to go further",
  animation_failed: "Diagnosed, but the animation didn't build",
  failed: "This one did not finish",
};

/* SQLite writes `datetime('now')`, which is UTC with no zone marker, so it has
 * to be told it is UTC or every timestamp reads as local and "just now" comes
 * out hours in the future. */
const relativeDay = (stamp) => {
  const then = new Date(`${String(stamp).replace(" ", "T")}Z`);
  if (Number.isNaN(then.getTime())) return "";
  const days = Math.floor((Date.now() - then.getTime()) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return then.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

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

const SUGGESTIONS = [
  "Why doesn't my rule work?",
  "Show me the part with actual numbers",
  "What should I do instead?",
];

function renderSession(sessionId) {
  mount("#tpl-session");
  const stageList = $("#stage-list");
  const done = new Set();
  let lastReason = null;

  const drawStages = (active) => {
    stageList.replaceChildren(...STAGES.map(([id, label]) => {
      const isDone = done.has(id);
      const li = el("li", isDone ? "is-done" : id === active ? "is-active" : "");
      li.append(el("span", "mark", isDone ? "✓" : id === active ? "▸" : "·"));
      const text = el("span");
      text.append(document.createTextNode(label));
      // The most recent stage's own reasoning, so the wait shows real work
      // rather than a spinner. Only the latest, so the panel stays bounded.
      if (lastReason && lastReason.stage === id) text.append(el("span", "why", lastReason.text));
      li.append(text);
      return li;
    }));
  };
  drawStages("s1_diagnose");

  let beats = [];
  let video = null;
  let activeBeat = null;

  /* The trailing fade claims "there is more past the edge", so it is only
   * applied when that is true. Re-measured on resize because the same beat
   * count overflows at one width and fits at another. */
  const markScrollable = () => {
    const rail = $("#rail");
    if (!rail) return;
    rail.parentElement.classList.toggle("is-scrollable", rail.scrollWidth > rail.clientWidth + 1);
  };
  window.addEventListener("resize", markScrollable);

  const drawRail = () => {
    const rail = $("#rail");
    rail.replaceChildren(...beats.map((b) => {
      const timed = b.start_s !== null && b.start_s !== undefined;
      const btn = el("button", "beat" + (b.targets_misconception ? " is-target" : ""));
      btn.type = "button";
      btn.dataset.beat = b.id;
      btn.disabled = !timed || !video;
      btn.append(el("span", "beat-title", b.title));
      btn.append(el("span", "beat-time", timed ? fmt(b.start_s) : "planned"));
      btn.setAttribute("aria-label", timed ? `Jump to ${fmt(b.start_s)}, ${b.title}` : `${b.title}, not yet rendered`);
      btn.onclick = () => seek(b.id);
      return btn;
    }));
    markActive(activeBeat);
    markScrollable();
  };

  const markActive = (beatId) => {
    document.querySelectorAll(".beat").forEach((node) => {
      const on = node.dataset.beat === beatId;
      node.classList.toggle("is-active", on);
      if (on) node.setAttribute("aria-current", "true");
      else node.removeAttribute("aria-current");
    });
  };

  // Only scroll when the active beat actually changes: ontimeupdate fires
  // several times a second and would otherwise fight the user's own scrolling.
  const setActive = (beatId) => {
    if (beatId === activeBeat) return;
    activeBeat = beatId;
    markActive(beatId);
    const node = beatId && $(`.beat[data-beat="${beatId}"]`);
    if (node) node.scrollIntoView({ block: "nearest", inline: "center" });
  };

  const seek = (beatId) => {
    const b = beats.find((x) => x.id === beatId);
    if (!b || b.start_s === null || !video) return;
    video.currentTime = b.start_s + 0.05;
    /* Playing matters as much as seeking: a beat's first frame is nearly empty
     * because its content is still being written on, so landing without
     * playing shows the student almost nothing. Autoplay policy can still
     * refuse, and it refuses less for muted media -- which costs nothing here,
     * since the render has no audio track at all. Try with sound (so the
     * controls do not show a muted speaker in browsers that allow it), then
     * fall back rather than leaving the player parked on a blank frame. */
    video.play().catch(() => {
      video.muted = true;
      video.play().catch(() => {});
    });
    setActive(beatId);
  };

  const showVideo = (url) => {
    const wrap = $("#player-wrap");
    wrap.classList.remove("is-empty");
    video = document.createElement("video");
    video.src = url; video.controls = true; video.playsInline = true;
    video.setAttribute("aria-label", "Your explanation");
    wrap.replaceChildren(video);
    video.ontimeupdate = () => {
      const t = video.currentTime;
      const current = beats.find((b) => b.start_s !== null && t >= b.start_s && t < b.end_s);
      setActive(current ? current.id : null);
    };

    /* The cooldown. While the narration plays, the microphone is closed: the
     * narration is a voice talking about the student's own mistake, so a live
     * recogniser hears the tutor and either finds a wake phrase in it or files
     * the explanation as the student's next question. Listening comes back a beat
     * after the audio stops, not immediately, so the room's echo does not land in
     * the next turn.
     *
     * `playing` rather than `play`, because `play` fires before audio is actually
     * coming out and the gap is enough for the first words to be heard. */
    video.addEventListener("playing", () => dictation.suspend());
    video.addEventListener("pause", releaseMic);
    video.addEventListener("ended", releaseMic);

    drawRail();
  };

  /* Two different things now hold the microphone closed: the animation's
   * narration, and the tutor's spoken reply. `suspended` inside voice.js is a
   * flag rather than a counter, so whichever of the two finished first would
   * reopen the microphone while the other was still talking -- straight into the
   * feedback loop the cooldown exists to prevent. Releasing therefore asks
   * whether anything is still audible, instead of assuming this was the only
   * reason the microphone was closed. */
  let replyAudio = null;
  const talking = () =>
    (video && !video.paused && !video.ended) || (replyAudio && !replyAudio.paused);
  const releaseMic = () => { if (!talking()) dictation.resume(); };

  /* Say the last reply out loud.
   *
   * Additive throughout: no text is sent up (the endpoint speaks the reply it
   * already stored, so it cannot be made to say anything else), and every failure
   * path leaves the written reply on screen, which was always the real answer.
   * A 503 here is the ordinary case when no voice is configured, so it is not
   * reported as an error.
   *
   * The microphone is closed *before* play rather than on the `playing` event.
   * For the video that ordering is a nicety; here the audio is a voice answering
   * a question about the student's mistake, in the same words they would use to
   * ask the next one, so a single frame of it reaching an open microphone is a
   * question the student did not ask. */
  const speakReply = async (source = "reply") => {
    let url = null;
    try {
      const res = await fetch(`/api/sessions/${sessionId}/speak?source=${source}`,
        { method: "POST" });
      if (!res.ok) return;
      url = URL.createObjectURL(await res.blob());
      replyAudio?.pause();
      replyAudio = new Audio(url);
      const done = () => { releaseMic(); URL.revokeObjectURL(url); };
      replyAudio.addEventListener("ended", done);
      replyAudio.addEventListener("error", done);
      dictation.suspend();
      await replyAudio.play();
    } catch {
      if (url) URL.revokeObjectURL(url);
      releaseMic();
    }
  };

  /* Voice input, wired to the composer.
   *
   * The transcript is written straight into the input as it arrives, so the
   * student watches the sentence assemble and can see a misrecognition before
   * it is sent rather than after. On a completed turn the question is submitted
   * for them: the words they said appear as their own message either way, so a
   * bad transcript is visible in the transcript rather than hidden behind a
   * confirmation step.
   *
   * The microphone is open before anyone presses anything, so the state has to be
   * legible without being asked for. Three things carry it: the button's shape
   * and fill, a labelled toggle beside the heading that says "listening" or "off"
   * in words, and the hint line under the composer. Colour alone would not do it,
   * which is why muted is a slashed microphone and not just a dimmer one. */
  const HINTS = {
    off: "Voice is off. Turn on Hey Astray, or press the microphone to ask once.",
    armed: "Listening for “Hey Astray”. Or press the microphone to ask now.",
    capturing: "Listening. Stop talking for a moment and I'll send it.",
    suspended: "Paused while the animation is playing.",
    blocked: "Microphone blocked. Allow it in your browser settings to ask by voice.",
    error: "Voice input stopped. Press the microphone to start it again.",
  };

  const WAKE_LABELS = {
    off: "Hey Astray: off",
    armed: "Hey Astray: listening",
    capturing: "Hey Astray: your turn",
    suspended: "Hey Astray: paused",
    blocked: "Microphone blocked",
    error: "Voice unavailable",
  };

  const mic = $("#mic-btn");
  const wake = $("#wake-toggle");
  // Whether always-on listening is wanted, remembered per browser. A student who
  // muted it should not have to mute it again on the next question.
  const wakeWanted = () => localStorage.getItem("astray.wake") !== "off";

  const dictation = window.createDictation({
    onInterim: (text) => { $("#chat-input").value = text; },
    onFinal: (text) => { $("#chat-input").value = ""; utterance(text); },
    /* The wake phrase landed. The video is paused defensively -- playback
     * suspends listening, so in the ordinary case nothing is playing by the time
     * we get here -- and a short blip confirms the phrase was heard, because
     * without it the student cannot tell whether to start talking.
     *
     * A spoken reply is stopped rather than left to finish. Saying the wake
     * phrase over the tutor is an interruption, and every other voice assistant
     * treats it as one; talking through the student's second question would also
     * be talking into an open microphone. */
    onWake: () => {
      if (video && !video.paused) video.pause();
      replyAudio?.pause();
      blip();
    },
    onState: (state) => {
      mic.classList.toggle("is-live", state === "capturing");
      mic.classList.toggle("is-armed", state === "armed");
      mic.classList.toggle("is-off", state === "off" || state === "error");
      mic.classList.toggle("is-suspended", state === "suspended");
      mic.disabled = state === "blocked" || $("#chat-input").disabled;
      mic.setAttribute("aria-label",
        state === "capturing" ? "Stop and send" : "Ask by voice");

      wake.setAttribute("aria-pressed", String(state === "armed" || state === "capturing"
        || state === "suspended"));
      wake.classList.toggle("is-on", state === "armed" || state === "capturing");
      wake.disabled = state === "blocked";
      $("#wake-label").textContent = WAKE_LABELS[state] || WAKE_LABELS.off;

      $("#chat-input").placeholder =
        state === "capturing" ? "Listening…" : "Why doesn't that work?";
      $("#chat-hint").textContent = HINTS[state] || HINTS.off;
    },
  });

  /* A short blip on wake, built rather than fetched so it costs no asset and no
   * request. Best-effort throughout: an AudioContext created without a user
   * gesture starts suspended, and if the browser refuses to resume it the
   * feature loses its chime and keeps its visual cue. Too short and too tonal to
   * be transcribed as speech by the microphone that is about to open. */
  let audio = null;
  const blip = () => {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      audio.resume?.();
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.0001, audio.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.06, audio.currentTime + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.12);
      osc.connect(gain).connect(audio.destination);
      osc.start();
      osc.stop(audio.currentTime + 0.13);
    } catch { /* no audio output; the visual cue carries it */ }
  };

  if (dictation.supported) {
    mic.hidden = false;
    wake.hidden = false;

    // The microphone button is still the direct route: it takes a question now,
    // wake phrase or not, which is what makes this survive a noisy room.
    mic.onclick = () => {
      if (dictation.state === "capturing") { dictation.finish(); return; }
      if (video && !video.paused) video.pause();
      dictation.capture();
    };

    wake.onclick = () => {
      const on = dictation.state === "armed" || dictation.state === "capturing"
        || dictation.state === "suspended";
      localStorage.setItem("astray.wake", on ? "off" : "on");
      if (on) dictation.mute();
      else dictation.arm();
    };

    // Escape abandons the turn rather than sending it, which is the one thing a
    // student wants the moment they realise the room is too loud.
    $("#chat-input").addEventListener("keydown", (e) => {
      if (e.key === "Escape" && dictation.state === "capturing") {
        e.preventDefault();
        dictation.cancel();
        $("#chat-input").value = "";
      }
    });
  }

  /* Chat opens on `diagnosis_ready`, not on the video. Waiting for the render
   * would leave the student with nothing to do for three minutes.
   *
   * Idle listening starts here rather than at page load, and only if the
   * microphone is already granted. Auto-arming into a permission prompt the
   * student did not ask for is the behaviour that makes an always-on microphone
   * feel like something done to them; when the answer is "prompt", the controls
   * wait for a click, which is also the gesture the browser wants. */
  const enableChat = () => {
    $("#chat-input").disabled = false;
    $("#chat-form button[type=submit]").disabled = false;
    if (dictation.supported && dictation.state !== "blocked") mic.disabled = false;
    document.querySelectorAll(".suggestion").forEach((s) => (s.disabled = false));
    $("#chat-hint").textContent = HINTS[dictation.state] || HINTS.off;
    if (dictation.supported && wakeWanted()) {
      dictation.permission().then((granted) => {
        if (granted === "granted" && dictation.state === "off") dictation.arm();
      });
    }
  };

  const suggestions = $("#suggestions");
  suggestions.replaceChildren(...SUGGESTIONS.map((q) => {
    const b = el("button", "suggestion", q);
    b.type = "button";
    b.disabled = true;
    b.onclick = () => ask(q);
    return b;
  }));

  /* Tutor replies are markdown. Render the small subset it actually uses --
   * paragraphs, bullets, bold, inline code and beat citations -- by building
   * nodes, never by assigning markup. */
  const INLINE = /(\[beat:b\d+\]|\*\*.+?\*\*|`[^`]+`)/g;

  const citation = (beatId) => {
    const b = beats.find((x) => x.id === beatId);
    const timed = b && b.start_s !== null && b.start_s !== undefined;
    const btn = el("button", "cite");
    btn.type = "button";
    btn.append(el("span", "play", "▶"));
    // A middot, not an em dash: the tutor's replies are sanitised of dashes
    // server-side, and the chips sit inside that prose.
    btn.append(document.createTextNode(b ? (timed ? `${fmt(b.start_s)} · ${b.title}` : b.title) : beatId));
    btn.setAttribute("aria-label", b && timed ? `Jump to ${fmt(b.start_s)}, ${b.title}` : `Moment: ${b ? b.title : beatId}`);
    btn.disabled = !timed || !video;
    btn.onclick = () => seek(beatId);
    return btn;
  };

  const renderInline = (text, into) => {
    for (const part of text.split(INLINE)) {
      if (!part) continue;
      const cite = part.match(/^\[beat:(b\d+)\]$/);
      if (cite) { into.append(citation(cite[1])); continue; }
      if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
        const strong = el("strong");
        renderInline(part.slice(2, -2), strong);
        into.append(strong);
        continue;
      }
      if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
        into.append(el("code", null, part.slice(1, -1)));
        continue;
      }
      into.append(document.createTextNode(part));
    }
  };

  const renderBody = (text, into) => {
    let list = null;
    let para = null;
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      const bullet = line.match(/^[-*]\s+(.*)$/);
      if (bullet) {
        para = null;
        if (!list) { list = el("ul"); into.append(list); }
        const li = el("li");
        renderInline(bullet[1], li);
        list.append(li);
        continue;
      }
      list = null;
      if (!line) { para = null; continue; }
      if (!para) { para = el("p"); into.append(para); }
      else para.append(document.createTextNode(" "));
      renderInline(line, para);
    }
  };

  const addMessage = (role, text) => {
    $("#chat-empty").classList.add("hidden");
    const wrap = el("div", `msg ${role}`);
    wrap.append(el("div", "who", role === "user" ? "You" : "Tutor"));
    const body = el("div", "body");
    if (role === "user") body.textContent = text;
    else renderBody(text, body);
    wrap.append(body);
    $("#messages").append(wrap);
    $("#messages").scrollTop = $("#messages").scrollHeight;
  };

  const ask = async (question, { spoken = false } = {}) => {
    const q = question.trim();
    if (!q || $("#chat-input").disabled) return;
    addMessage("user", q);
    try {
      const res = await api(`/api/sessions/${sessionId}/chat`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ message: q }),
      });
      addMessage("assistant", res.reply);
      // Answered aloud only when it was asked aloud. A student who typed is
      // reading, and reading is faster than listening.
      if (spoken) speakReply();
    } catch (e2) {
      addMessage("assistant", `Unable to answer right now: ${e2.message}. Try again in a moment.`);
    }
  };

  /* Everything spoken goes through here, and this is where "hey Astray" stopped
   * meaning "ask a question about this animation".
   *
   * A student mid-session says two quite different kinds of thing. "I still don't
   * get the binomial thing" belongs in this conversation. "I was solving dy/dx
   * equals 2x minus 5 over y squared and I couldn't figure it out" is a different
   * problem, and answering it in this chat was the failure: the reply arrived
   * grounded in an animation about expanding a bracket, and no animation was ever
   * built for the question actually asked.
   *
   * The server decides which, because deciding needs the maths read properly and
   * the recogniser does not deliver maths -- it delivered "DUI over DX" for
   * dy/dx. The same call that classifies also repairs the notation, so a new
   * problem arrives as `dy/dx = (2x-5)/y^2` rather than as a sentence about DUI.
   *
   * The hint line carries the wait. Routing costs a couple of seconds, and a
   * student who has just finished talking to an interface that has gone silent
   * assumes it did not hear them. */
  const utterance = async (text) => {
    const said = text.trim();
    if (!said) return;
    const hint = $("#chat-hint");
    hint.textContent = "Working out what you're asking…";
    let routed;
    try {
      routed = await api("/api/voice/route", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ transcript: said, session_id: sessionId }),
      });
    } catch {
      // Routing is a convenience over "send it to the tutor". Losing it must not
      // lose the sentence the student just said.
      routed = { kind: "followup", question: said };
    }
    hint.textContent = HINTS[dictation.state] || HINTS.off;

    if (routed.kind === "new_problem" && routed.problem) {
      await startProblem(routed);
      return;
    }
    ask(routed.question || said, { spoken: true });
  };

  /* A new problem, spoken. Creating the session and changing the hash is all it
   * takes to start the pipeline: the session view opens the stream, and opening
   * the stream is what runs the chain. So this is the same path the typed form
   * takes, reached by talking.
   *
   * `astray.spoke` survives the navigation. The new session's own chat should
   * answer aloud too, because the student asked for this one out loud and has not
   * touched the keyboard. */
  const startProblem = async (routed) => {
    addMessage("user", routed.problem);
    addMessage("assistant", `That's a different problem, so I'm building you a new animation for `
      + `it. One moment.`);
    try {
      const { session_id } = await api("/api/sessions", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ handle, problem: routed.problem, work: routed.work || "" }),
      });
      sessionStorage.setItem("astray.spoke", session_id);
      location.hash = `#/session/${session_id}`;
    } catch (e2) {
      addMessage("assistant", `I heard a new problem but couldn't start it: ${e2.message}.`);
    }
  };

  $("#chat-form").onsubmit = (e) => {
    e.preventDefault();
    const input = $("#chat-input");
    const q = input.value;
    /* Enter during a turn sends what is in the box now. The turn has to be
     * abandoned rather than finished, or the recogniser's own commit sends the
     * same sentence a second time. */
    if (dictation.state === "capturing") dictation.cancel();
    input.value = "";
    ask(q);
  };

  /* A session created by speaking answers its first question out loud without
   * being asked to. The student spoke a problem into a machine and has not
   * touched the keyboard, so a silent diagnosis is the same "it did not respond
   * at all" that the written reply already looked like. Read from sessionStorage,
   * because this session was navigated to, not asked for in place. Announced at
   * most once: showDiagnosis runs both for the live event and for a reload of a
   * finished session. */
  const spokeThisIn = sessionStorage.getItem("astray.spoke") === sessionId;
  sessionStorage.removeItem("astray.spoke");
  let announced = false;

  /* Whether this session has any working to diagnose, learned from the session
   * fetch below. That fetch resolves in milliseconds and the diagnosis takes 15 to
   * 30 seconds, so it is always known by the time the card is drawn; on a reload of
   * a finished session both arrive from the same request, in order. */
  let explaining = false;

  const showDiagnosis = async (d) => {
    const card = $("#diagnosis");
    card.classList.remove("is-pending");
    card.replaceChildren();
    // A session with no working has no mistake to announce, and its diagnosis text
    // is an apology for not finding one. Speak the method instead.
    if (spokeThisIn && !announced) {
      announced = true;
      speakReply(explaining ? "solution" : "diagnosis");
    }

    if (d.no_error_found) {
      card.append(el("p", "eyebrow", "Diagnosis"));
      card.append(el("h1", null, "Your working is correct"));
      card.append(el("p", null, d.misconception_statement));
      return;
    }

    /* Nothing was attempted, so there is nothing to diagnose. The card that says
     * "here's where it went astray" would be accusing them of a mistake they never
     * made, and the statement the diagnosis carries here is an apology for not
     * finding one. What it does carry is a solved, SymPy-checked solution, so that
     * is what gets shown, and the animation teaches the method. */
    if (explaining) {
      card.append(el("p", "eyebrow", "How to solve it"));
      card.append(el("h1", null, "Here's how this one works."));
      const steps = el("ol", "solution");
      (d.correct_solution || []).forEach((s) => steps.append(el("li", "mono", s)));
      if (steps.childElementCount) card.append(steps);
      const badges = el("div", "badges");
      if (d.verified_by_sympy) badges.append(el("span", "badge is-good", "✓ checked with SymPy"));
      badges.append(el("span", "badge", "no working shown, so nothing to correct"));
      card.append(badges);
      return;
    }

    card.append(el("p", "eyebrow", "Diagnosis"));
    card.append(headline());
    card.append(el("code", "rule", d.buggy_rule));
    card.append(el("p", null, d.misconception_statement));

    const badges = el("div", "badges");
    const badge = (text, cls = "") => badges.append(el("span", `badge ${cls}`.trim(), text));
    if (d.verified_by_sympy) badge("✓ checked with SymPy", "is-good");
    else badge("not symbolically checkable");
    if (d.divergence_index !== null && d.divergence_index !== undefined)
      badge(`diverges at step ${d.divergence_index + 1}`);
    badge(`confidence ${Math.round(d.confidence * 100)}%`);
    card.append(badges);

    try {
      const peers = await api(`/api/sessions/${sessionId}/peers`);
      if (peers.others > 0)
        badge(`${peers.others} other student${peers.others > 1 ? "s" : ""} made this error`, "is-peers");
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
    explaining = Boolean(session.explaining);
    if (session.problem) {
      $("#problem-text").textContent = session.problem;
      $("#problem-line").hidden = false;
    }
    if (session.diagnosis) { showDiagnosis(session.diagnosis); enableChat(); }
    beats = info.beats;
    drawRail();
    if (info.video_url) showVideo(info.video_url);
    const history = await api(`/api/sessions/${sessionId}/chat`).catch(() => ({ messages: [] }));
    history.messages.forEach((m) => addMessage(m.role, m.content));
  })().catch((e) => console.error("session init failed", e));

  const es = new EventSource(`/api/sessions/${sessionId}/stream`);
  es.addEventListener("stage_started", (e) => drawStages(JSON.parse(e.data).stage));
  es.addEventListener("stage_completed", (e) => {
    const ev = JSON.parse(e.data);
    done.add(ev.stage);
    if (ev.payload?.reasoning) lastReason = { stage: ev.stage, text: ev.payload.reasoning };
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
    try {
      const message = JSON.parse(e.data).message || "Something went wrong building the animation.";
      const box = $("#pipeline");
      box.replaceChildren(el("p", "eyebrow", "Animation"), el("p", "field-hint", message));
    } catch { /* connection blip, not a server-sent error event */ }
  });
  es.addEventListener("done", () => es.close());
}

/* ----------------------------------------------------------------- insights */
async function renderInsights() {
  mount("#tpl-insights");

  const empty = (title, body, cta) => {
    const box = el("div", "empty");
    box.append(el("p", "empty-title", title));
    box.append(el("p", "field-hint", body));
    if (cta) {
      const a = el("a", "btn btn-quiet", cta);
      a.href = "#/";
      box.append(a);
    }
    return box;
  };

  const statList = (items, label) => {
    const max = Math.max(1, ...items.map((m) => m.value));
    // A bar is a comparison. With a single row there is nothing to compare it
    // against, and a lone full-width bar reads as a rule under the text.
    const comparable = items.length > 1;
    const list = el("div", "stat-list");
    items.forEach((m) => {
      const row = el("div", "stat");
      row.append(el("span", "stat-label", m.canonical_statement));
      row.append(el("span", "stat-count", label(m.value)));
      if (comparable) {
        const bar = el("span", "stat-bar");
        bar.style.width = `${Math.round((m.value / max) * 100)}%`;
        row.append(bar);
      }
      list.append(row);
    });
    return list;
  };

  try {
    const data = await api(`/api/insights?handle=${encodeURIComponent(handle)}`);

    $("#history").replaceChildren(
      data.history.length
        ? statList(data.history.map((m) => ({ ...m, value: m.times })), (n) => `${n}×`)
        : empty("No patterns yet",
            "Your own misconceptions show up here once you have been diagnosed, so repeats are easy to spot.",
            "Diagnose some work"));

    $("#frequency").replaceChildren(
      data.misconceptions.length
        ? statList(data.misconceptions.map((m) => ({ ...m, value: m.students })),
            (n) => `${n} student${n > 1 ? "s" : ""}`)
        : empty("No diagnoses logged yet",
            "Every diagnosis anyone runs lands here as an anonymous count."));
  } catch (e) {
    $("#history").replaceChildren(empty("Unable to load insights", e.message));
    $("#frequency").replaceChildren();
  }
}
