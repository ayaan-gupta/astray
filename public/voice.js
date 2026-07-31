/* Dictation for the chat composer, always-on with a wake phrase.
 *
 * What the browser gives us is one recogniser. What this builds on top is two
 * modes for it, because "always listening" and "taking a question" want opposite
 * things from the same stream of words:
 *
 *   armed      The recogniser is running and everything it hears is thrown away,
 *              except that each result is scanned for the wake phrase. Nothing
 *              reaches the composer and nothing is sent.
 *   capturing  Exactly the turn this file already took: accumulate, show the
 *              transcript live, end on a gap of silence, submit.
 *
 * Only the entry into `capturing` is new. Everything after it -- the silence
 * gap, `stop()` rather than `abort()` so the tail of the sentence survives, the
 * permission handling, the commit -- is the push-to-talk turn unchanged, which is
 * why the microphone button still works as a button. A wake phrase is a second
 * doorway into the same room.
 *
 * There is no always-on API. `continuous` is a request, not a promise: Chrome
 * ends a session on its own after roughly a minute, and sooner on quiet, firing
 * `end` as though the speaker had finished. Restarting from `end` *is* the
 * always-on mechanism, and it is the reason this file needs a fast-fail guard --
 * an `end` that arrives immediately, forever, is an infinite loop with a
 * microphone in it.
 *
 * Two honest limits, both structural rather than fixable here. The recogniser
 * streams audio to Google's servers, so idle listening means an open network
 * connection carrying whatever is said near the machine; that is the cost of
 * always-on on this API and the reason muting is a first-class control rather
 * than a preference buried somewhere. And detection is edit distance on a
 * general transcript (see `wake.js`), not a trained keyword model, so it misses
 * and misfires at a rate a real assistant would not.
 */

/* The gap that ends a turn. Long enough to survive the pause before the second
 * half of "why doesn't that work ... for negative numbers", short enough that
 * the student is not left waiting once they have finished. */
const SILENCE_MS = 1600;

/* The wake phrase and the question are usually one breath -- "hey astray why
 * doesn't that work" -- but not always. When the phrase arrives with nothing
 * behind it the turn waits longer than a normal gap before giving up, because
 * the student is at that moment deciding what to ask. */
const FIRST_WORD_MS = 4000;

// A single captured turn never runs longer than this, whatever the recogniser does.
const MAX_MS = 30000;

/* How long after the tutor stops talking before idle listening resumes. Covers
 * the tail of a speaker and the room's own echo of it. */
const COOLDOWN_MS = 800;

/* Ambient transcript kept while armed. Only enough to hold a wake phrase and
 * whatever it arrived in the middle of; without a cap this grows for as long as
 * the page is open. */
const WAKE_WINDOW = 160;

/* An `end` sooner than this after a start means the session never really ran.
 * A handful in a row is a recogniser that cannot work -- no device, no network,
 * a revoked permission -- and idle listening stops rather than spinning. */
const FAST_FAIL_MS = 400;
const FAST_FAIL_LIMIT = 5;

/* Chrome throws if `start()` is called while a session is still winding down, so
 * restarts go through a tick of breathing room. It also stops a fast-fail loop
 * from being a busy loop. */
const RESTART_DELAY_MS = 180;

const squash = (...parts) => parts.join(" ").replace(/\s+/g, " ").trim();

/* States, each of which the UI shows differently:
 *   unsupported  no SpeechRecognition, or an insecure origin. The controls hide.
 *   off          muted. Nothing is running and nothing is being heard.
 *   armed        idle-listening for the wake phrase.
 *   capturing    taking a question, with a live transcript in the composer.
 *   suspended    armed, but stood down while the tutor's own audio plays.
 *   blocked      the microphone was refused. Terminal for the page: the prompt
 *                cannot be raised again from script.
 *   error        the recogniser failed for some other reason. Recoverable by
 *                arming again.
 */
function createDictation({
  onInterim = () => {},
  onFinal = () => {},
  onState = () => {},
  onWake = () => {},
} = {}) {
  const Impl = window.SpeechRecognition || window.webkitSpeechRecognition;

  /* A secure context is a hard requirement, not a nicety: on plain http the
   * constructor exists and `start()` fails at the permission step, which would
   * present the student with controls that cannot ever work. */
  const supported = Boolean(Impl) && window.isSecureContext;

  let rec = null;
  let mode = "off"; // off | armed | capturing
  let suspended = false;
  let terminal = null; // "blocked" | "error", once and for the page
  let running = false; // a session is believed to be open
  let finishing = false; // stop() sent; waiting for the final result and `end`
  let woke = false; // this turn began at a wake phrase, not the button
  let settled = ""; // transcript the recogniser has committed to
  let pending = ""; // its current best guess at what is still being said
  let silenceTimer = null;
  let capTimer = null;
  let cooldownTimer = null;
  let restartTimer = null;
  let startedAt = 0;
  let fastFails = 0;
  let reported = null;

  const publicState = () => {
    if (!supported) return "unsupported";
    if (terminal) return terminal;
    if (mode === "off") return "off";
    if (suspended) return "suspended";
    return mode;
  };

  const report = () => {
    const next = publicState();
    if (next === reported) return;
    reported = next;
    onState(next);
  };

  const clearTurnTimers = () => {
    clearTimeout(silenceTimer);
    clearTimeout(capTimer);
    silenceTimer = capTimer = null;
  };

  /* The question, with the wake phrase and anything before it removed.
   *
   * This re-runs the match on every result rather than remembering where the
   * phrase was, and that is deliberate. The recogniser revises interim text --
   * "hey a stray" becomes "hey astray" as it gets more audio, which changes the
   * length of the prefix and would invalidate a stored index. Re-finding it is
   * self-correcting. If a revision loses the phrase altogether the whole
   * transcript is used, which leaves "hey astray" at the front of the question:
   * the tutor can read past that, and it is a better failure than dropping the
   * question. */
  const question = (text) => {
    if (!woke) return text;
    const at = window.findWake(text);
    return at >= 0 ? text.slice(at).trim() : text;
  };

  const armSilence = (ms) => {
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(finish, ms);
  };

  /* Hand the transcript over and go back to idle listening.
   *
   * The old version of this returned to a single idle state. Now the resting
   * place is `armed`, so a student can ask a second question without touching
   * anything -- which is the entire point of the change. */
  function commit() {
    const wasCapturing = mode === "capturing";
    finishing = false;
    clearTurnTimers();
    if (!wasCapturing) return;

    const text = question(squash(settled, pending));
    settled = pending = "";
    woke = false;
    mode = terminal ? "off" : "armed";
    report();
    onInterim("");
    if (text) onFinal(text);
    ensureRunning();
  }

  // Ask for the last of the audio, then wait for `end` to deliver it.
  function finish() {
    if (mode !== "capturing" || finishing) return;
    finishing = true;
    clearTurnTimers();
    try {
      rec.stop();
    } catch {
      commit();
    }
  }

  /* Enter the capture turn. `rest` is whatever the student said after the wake
   * phrase in the same breath, which is usually the whole question. */
  const beginCapture = (rest, fromWake) => {
    // Belt and braces behind the guard in `onresult`: nothing may enter a capture
    // turn while the tutor's audio is playing or the microphone is muted.
    if (suspended || terminal || (mode === "off" && fromWake)) return;
    mode = "capturing";
    woke = Boolean(fromWake);
    settled = rest ? `${rest.trim()} ` : "";
    pending = "";
    finishing = false;
    report();
    if (fromWake) onWake();
    onInterim(squash(settled));
    armSilence(rest ? SILENCE_MS : FIRST_WORD_MS);
    capTimer = setTimeout(finish, MAX_MS);
  };

  const build = () => {
    const r = new Impl();
    r.lang = "en-US";
    r.continuous = true;
    r.interimResults = true;
    r.maxAlternatives = 1;

    r.onresult = (event) => {
      /* Deaf while stood down, and this guard is the difference between the
       * cooldown working and merely looking like it works.
       *
       * `abort()` does not retract what is already in flight: a result the
       * recogniser had queued before we suspended still arrives, and the handler
       * runs with `suspended` already true. Without this line that result is
       * scanned for the wake phrase like any other -- so the narration saying
       * "find where your reasoning went astray" could wake the assistant and
       * hand it the tutor's own sentence as the student's question, which is the
       * exact failure the cooldown exists to prevent. Caught by a stub test that
       * delivered a result after `suspend()`; the wake fired and the UI went on
       * claiming it was paused, because `suspended` masks the mode it reports. */
      if (suspended || terminal || mode === "off") {
        pending = "";
        return;
      }

      /* `resultIndex` is where this event's news starts; everything before it
       * has already been folded into `settled`. Interim results are rewritten on
       * every event, so `pending` is rebuilt rather than appended to. */
      pending = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) settled += chunk;
        else pending += chunk;
      }
      fastFails = 0; // words are arriving, so the recogniser is healthy

      const text = squash(settled, pending);

      if (mode === "capturing") {
        const asked = question(text);
        onInterim(asked);
        // Still waiting for the first word after the phrase: keep the longer
        // grace period rather than cutting them off while they think.
        armSilence(asked ? SILENCE_MS : FIRST_WORD_MS);
        return;
      }

      if (mode !== "armed") return;

      const at = window.findWake(text);
      if (at >= 0) {
        beginCapture(text.slice(at), true);
        return;
      }
      // Nothing addressed to us. Keep only enough to catch a phrase that
      // straddles two results.
      if (settled.length > WAKE_WINDOW) settled = settled.slice(-WAKE_WINDOW);
    };

    r.onerror = (event) => {
      // Both of these arrive with an `end` immediately behind them, which is
      // where they are already handled.
      if (event.error === "no-speech" || event.error === "aborted") return;

      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        terminal = "blocked";
        mode = "off";
        finishing = false;
        clearTurnTimers();
        report();
        return;
      }
      /* Everything else -- `network` above all -- is transient, and on an
       * always-on listener a transient failure must not be terminal or a single
       * dropped connection ends listening for the rest of the page. `end` will
       * follow and the restart path treats it like any other, with the fast-fail
       * guard as the backstop if it keeps happening. */
    };

    r.onend = () => {
      running = false;

      if (finishing) {
        commit();
        return;
      }
      if (terminal || mode === "off" || suspended) return;

      /* Ended by itself. This is the always-on mechanism: the session is over
       * and the student has said nothing to end it, so open another one. */
      if (Date.now() - startedAt < FAST_FAIL_MS) {
        fastFails += 1;
        if (fastFails >= FAST_FAIL_LIMIT) {
          terminal = "error";
          mode = "off";
          clearTurnTimers();
          report();
          return;
        }
      } else {
        fastFails = 0;
      }
      ensureRunning();
    };

    return r;
  };

  function ensureRunning() {
    if (!supported || terminal || suspended || mode === "off" || running) return;
    clearTimeout(restartTimer);
    restartTimer = setTimeout(() => {
      if (!supported || terminal || suspended || mode === "off" || running) return;
      if (!rec) rec = build();
      try {
        rec.start();
        running = true;
        startedAt = Date.now();
      } catch {
        /* Already winding down. `end` is still coming and will bring us back
         * here, so this is a missed beat rather than a lost listener. */
      }
    }, RESTART_DELAY_MS);
  }

  const stopRunning = () => {
    clearTimeout(restartTimer);
    if (!rec) return;
    try {
      rec.abort();
    } catch { /* never started, or already dead */ }
    running = false;
  };

  /* ------------------------------------------------------------------ public */

  // Turn idle listening on. Also the recovery path out of `error`.
  const arm = () => {
    if (!supported || terminal === "blocked") return;
    terminal = null;
    fastFails = 0;
    suspended = false;
    clearTimeout(cooldownTimer);
    settled = pending = "";
    if (mode === "off") mode = "armed";
    report();
    ensureRunning();
  };

  const mute = () => {
    if (!supported) return;
    mode = "off";
    suspended = false;
    finishing = false;
    woke = false;
    settled = pending = "";
    clearTurnTimers();
    clearTimeout(cooldownTimer);
    stopRunning();
    report();
    onInterim("");
  };

  // The microphone button: take a question now, wake phrase or not.
  const capture = () => {
    if (!supported || terminal === "blocked" || mode === "capturing") return;
    terminal = null;
    suspended = false;
    clearTimeout(cooldownTimer);
    settled = pending = "";
    beginCapture("", false);
    ensureRunning();
  };

  // Abandon a turn without sending it, and fall back to idle listening.
  const cancel = () => {
    if (mode !== "capturing") return;
    finishing = false;
    woke = false;
    settled = pending = "";
    clearTurnTimers();
    mode = "armed";
    /* The session is aborted rather than left open: it is mid-utterance with a
     * half-spoken question in it, and `armed` should start from silence. `end`
     * brings idle listening back up. */
    stopRunning();
    report();
    onInterim("");
    ensureRunning();
  };

  /* Stand down while the tutor's own audio plays.
   *
   * This is the feedback problem the feature cannot ship without. The narration
   * is a voice explaining the student's mistake, and every sentence of it is
   * about the very thing the student would ask about -- so a live microphone
   * hears the tutor, transcribes it, and either finds a wake phrase in it or
   * files the tutor's own explanation as the student's next question. Muting the
   * element is not enough on a laptop with open speakers; the recogniser has to
   * be closed.
   *
   * A capture in flight is abandoned rather than finished, because audio starting
   * mid-question means whatever was captured is already polluted. */
  const suspend = () => {
    if (!supported || terminal || mode === "off") return;
    clearTimeout(cooldownTimer);
    if (mode === "capturing") {
      finishing = false;
      woke = false;
      clearTurnTimers();
      mode = "armed";
      onInterim("");
    }
    if (suspended) return;
    suspended = true;
    settled = pending = "";
    stopRunning();
    report();
  };

  const resume = (delay = COOLDOWN_MS) => {
    if (!supported || terminal || mode === "off" || !suspended) return;
    clearTimeout(cooldownTimer);
    cooldownTimer = setTimeout(() => {
      if (!suspended) return;
      suspended = false;
      // Anything captured through the speaker is not the student talking.
      settled = pending = "";
      report();
      ensureRunning();
    }, delay);
  };

  /* Whether the microphone is already granted, so the page can decide between
   * arming on load and waiting for a click. Auto-arming into an unexpected
   * permission prompt is precisely the behaviour that makes an always-on
   * microphone feel like something done to the student rather than for them.
   * Firefox has no `microphone` descriptor and throws; unknown means ask. */
  const permission = async () => {
    if (!supported) return "unsupported";
    try {
      const status = await navigator.permissions.query({ name: "microphone" });
      return status.state;
    } catch {
      return "unknown";
    }
  };

  report();

  return {
    get supported() { return supported; },
    get state() { return publicState(); },
    arm,
    mute,
    capture,
    finish,
    cancel,
    suspend,
    resume,
    permission,
  };
}

window.createDictation = createDictation;
