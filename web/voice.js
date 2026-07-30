/* Dictation for the chat composer. Push to talk, not a wake word.
 *
 * The browser's SpeechRecognition is the whole engine here, and it is worth
 * being precise about what it is: general speech-to-text streamed to Google's
 * servers, Chromium only, with no local model and no purpose-built keyword
 * spotter. That is fine for "press the button and ask a question" and it is not
 * fine for always-on listening, which is why this is a button. A wake phrase on
 * top of this API is a fuzzy string match against a general transcript -- it
 * misfires on "hey a stray" and misses in a noisy room -- and the honest way to
 * build one is a dedicated on-device keyword model. The capture half below is
 * the same either way, so that route stays open.
 *
 * Three things make this more than a call to `.start()`:
 *
 * 1. A continuous session is not continuous. Chrome ends one on its own after
 *    roughly a minute, and sooner on a stretch of quiet, firing `end` as though
 *    the speaker had finished. So `end` means "restart" while the student is
 *    still holding the floor, and only means "finished" when we asked it to
 *    stop. The restart is counted, because a recogniser that cannot start is
 *    otherwise an infinite loop.
 *
 * 2. A turn ends on a silence gap, not on a timer. A fixed timeout either cuts
 *    the student off mid-sentence or leaves them waiting after they have
 *    finished; a gap measured from the last speech event ends the turn when
 *    they actually stop talking.
 *
 * 3. `stop()` and `abort()` are not interchangeable. `stop()` finalises the
 *    audio already captured and delivers one last result; `abort()` throws it
 *    away. Ending a turn has to use `stop()` and wait for `end`, or the last
 *    few words the student said are lost -- which is exactly the tail of the
 *    question, where the actual question usually is.
 */

/* The gap that ends a turn. Long enough to survive the pause before the second
 * half of "why doesn't that work ... for negative numbers", short enough that
 * the student is not left waiting once they have finished. */
const SILENCE_MS = 1600;

// A single dictation never runs longer than this, whatever the recogniser does.
const MAX_MS = 30000;

/* How many times a turn may be resumed after the recogniser ends by itself.
 * Eight covers any question a student will ask; the cap is here so a recogniser
 * that ends immediately and forever cannot spin. */
const RESTART_LIMIT = 8;

const squash = (...parts) => parts.join(" ").replace(/\s+/g, " ").trim();

/* States, all four of which the UI shows differently:
 *   unsupported  no SpeechRecognition, or an insecure origin. The control hides.
 *   idle         ready; the mic is offered.
 *   listening    capturing, with a live transcript in the composer.
 *   blocked      the student refused the microphone, or the browser did. This is
 *                terminal for the page: the prompt cannot be raised again from
 *                script, so re-offering the button would be a dead control.
 *   error        the recogniser failed for some other reason. Recoverable.
 */
function createDictation({ onInterim = () => {}, onFinal = () => {}, onState = () => {} } = {}) {
  const Impl = window.SpeechRecognition || window.webkitSpeechRecognition;

  /* A secure context is a hard requirement, not a nicety: on plain http the
   * constructor exists and `start()` fails at the permission step, which would
   * present the student with a mic button that cannot ever work. */
  const supported = Boolean(Impl) && window.isSecureContext;

  let rec = null;
  let capturing = false; // the student has the floor
  let finishing = false; // stop() sent; waiting for the final result and `end`
  let settled = "";      // transcript the recogniser has committed to
  let pending = "";       // its current best guess at what is still being said
  let silenceTimer = null;
  let capTimer = null;
  let restarts = 0;
  let state = supported ? "idle" : "unsupported";

  const setState = (next) => {
    if (next === state) return;
    state = next;
    onState(state);
  };

  const clearTimers = () => {
    clearTimeout(silenceTimer);
    clearTimeout(capTimer);
    silenceTimer = capTimer = null;
  };

  /* Hand the transcript over and go back to idle.
   *
   * `wasActive` is what keeps this from trampling a terminal state. Every path
   * out of a turn ends in `end`, including a refused microphone, so this runs
   * after `blocked` has already been set -- and must leave it alone rather than
   * reporting the recogniser as idle and ready. */
  const commit = () => {
    const wasActive = capturing || finishing;
    capturing = false;
    finishing = false;
    restarts = 0;
    clearTimers();
    if (!wasActive) return;

    const text = squash(settled, pending);
    settled = pending = "";
    setState("idle");
    onInterim("");
    if (text) onFinal(text);
  };

  // Ask for the last of the audio, then wait for `end` to deliver it.
  const finish = () => {
    if (!capturing || finishing) return;
    finishing = true;
    clearTimers();
    try {
      rec.stop();
    } catch {
      commit();
    }
  };

  const armSilence = () => {
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(finish, SILENCE_MS);
  };

  const build = () => {
    const r = new Impl();
    r.lang = "en-US";
    r.continuous = true;
    r.interimResults = true;
    r.maxAlternatives = 1;

    r.onresult = (event) => {
      /* `resultIndex` is where this event's news starts; everything before it
       * has already been folded into `settled`. Interim results are rewritten
       * on every event, so `pending` is rebuilt rather than appended to. */
      pending = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) settled += chunk;
        else pending += chunk;
      }
      onInterim(squash(settled, pending));
      armSilence();
    };

    r.onerror = (event) => {
      // Both of these arrive with an `end` immediately behind them, which is
      // where the turn is already handled.
      if (event.error === "no-speech" || event.error === "aborted") return;

      capturing = false;
      finishing = false;
      clearTimers();
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        setState("blocked");
        return;
      }
      setState("error");
    };

    r.onend = () => {
      if (finishing || !capturing) {
        commit();
        return;
      }
      // Ended by itself with the student still talking: pick the turn back up.
      if (restarts >= RESTART_LIMIT) {
        finishing = true;
        commit();
        return;
      }
      restarts += 1;
      try {
        r.start();
      } catch {
        finishing = true;
        commit();
      }
    };

    return r;
  };

  const start = () => {
    if (!supported || capturing || finishing) return;
    if (state === "blocked") return;

    settled = pending = "";
    restarts = 0;
    capturing = true;
    finishing = false;
    rec = build();
    try {
      rec.start();
    } catch {
      capturing = false;
      setState("error");
      return;
    }
    setState("listening");
    armSilence();
    capTimer = setTimeout(finish, MAX_MS);
  };

  // Abandon the turn without sending anything. Escape, and losing focus.
  const cancel = () => {
    if (!capturing && !finishing) return;
    capturing = false;
    finishing = false;
    clearTimers();
    settled = pending = "";
    try {
      rec.abort();
    } catch { /* already dead; nothing to abandon */ }
    setState("idle");
    onInterim("");
  };

  const toggle = () => (capturing || finishing ? finish() : start());

  return {
    get supported() { return supported; },
    get state() { return state; },
    start,
    finish,
    cancel,
    toggle,
  };
}

window.createDictation = createDictation;
