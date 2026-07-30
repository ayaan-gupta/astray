/* Finding "hey astray" in a transcript that will rarely contain those letters.
 *
 * Split out of `voice.js` because this is the one part of the feature that is
 * pure string work, and therefore the one part that can be tested exhaustively
 * without a recogniser, a microphone, or a browser at all.
 *
 * An exact match is the wrong tool. `SpeechRecognition` is general-purpose
 * speech-to-text with no notion of a keyword, so it transcribes the product's
 * name as whatever ordinary English it sounds closest to: "hey a stray", "hey
 * astro", "hey ashtray", "hey stray". Every one of those is the student saying
 * the wake phrase correctly and the recogniser writing it down differently.
 *
 * So the match is edit distance against the name, with two rules that between
 * them keep the false-positive rate down:
 *
 *   1. A trigger word is required. "astray" appears constantly in this product's
 *      own copy -- it is the name and the verb in the tagline -- so matching the
 *      bare word would fire every time the tutor's narration said it. Requiring
 *      "hey" in front means the phrase has to be addressed to something.
 *
 *   2. Candidates must be at least four characters. Edit distance two on a short
 *      token matches far too much; "a", "at" and "as" are all within two of
 *      "astray" if you let them be.
 *
 * The real fix for wake-word detection is a small on-device model trained on the
 * phrase -- Porcupine, openWakeWord -- which is a different dependency, an
 * access key, and a trained keyword file. This is the honest version of what can
 * be built on a general transcript, and it will occasionally miss and
 * occasionally misfire. Both failures are recoverable by hand: the microphone
 * button still starts a turn directly.
 */

// Words that make the phrase an address rather than a mention.
const TRIGGERS = new Set(["hey", "hi", "hey,", "ok", "okay", "hello", "yo"]);

const NAME = "astray";

/* Two edits. "astro" and "a tray" are both two away from the name, and both are
 * things Chrome actually returns for someone saying it correctly. Three starts
 * pulling in unrelated words. */
const MAX_EDITS = 2;

// Below this, edit distance stops discriminating: most short words are within
// two of most other short words.
const MIN_CANDIDATE = 4;

/* How many words after the trigger may be joined to form a candidate. Two,
 * because the recogniser's favourite mistake is splitting the name in half:
 * "a stray", "as tray", "ash tray". */
const MAX_JOIN = 2;

/** Levenshtein distance, iterative with a single row. */
function editDistance(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;

  let row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    let previous = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const carried = row[j];
      row[j] = Math.min(
        row[j] + 1, // deletion
        row[j - 1] + 1, // insertion
        previous + (a[i - 1] === b[j - 1] ? 0 : 1), // substitution
      );
      previous = carried;
    }
  }
  return row[b.length];
}

/**
 * Where the wake phrase ends in `text`, or -1 if it is not there.
 *
 * The return value is an index into the string that was passed in, so the caller
 * can slice off everything up to and including the phrase and keep the rest as
 * the question: "hey astray why doesn't that work" leaves "why doesn't that
 * work". `toLowerCase` preserves length for the Latin text a transcript
 * contains, so indices taken from the lowered copy are valid in the original.
 */
function findWake(text) {
  const low = String(text).toLowerCase();
  const words = [...low.matchAll(/[a-z']+/g)];

  for (let i = 0; i < words.length; i += 1) {
    if (!TRIGGERS.has(words[i][0])) continue;

    for (let take = 1; take <= MAX_JOIN; take += 1) {
      const parts = words.slice(i + 1, i + 1 + take);
      if (parts.length < take) break;

      // Joined without the space, because the space is the recogniser's
      // invention: "a stray" and "astray" are the same sound.
      const candidate = parts.map((p) => p[0]).join("");
      if (candidate.length < MIN_CANDIDATE) continue;
      if (editDistance(candidate, NAME) > MAX_EDITS) continue;

      const last = parts[parts.length - 1];
      return last.index + last[0].length;
    }
  }
  return -1;
}

window.findWake = findWake;
window.editDistance = editDistance;
