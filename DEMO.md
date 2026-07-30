# Demo notes

Everything below is already in `data/tutor.db`. Nothing here needs the pipeline
to run, so the demo does not depend on a live model call.

```bash
uv run uvicorn server.app:create_app --factory --port 8000
```

## The golden case

**`(y + 3)^2` → `y^2 + 9`** — session `fa8f9e92-220d-4345-b9e5-484a228fb6ff`

<http://localhost:8000/#/session/fa8f9e92-220d-4345-b9e5-484a228fb6ff>

Status `ready`, 38.0s narrated video, four beats. The beats were timed by the
container's own clock and then each was held open for as long as its spoken line
needed (see *The video fits the explanation* below); these are the published
timings, which are what the rail seeks to:

| Beat | Title | Primitive | Start |
|---|---|---|---|
| b1 | Correct expansion | `algebra_steps` | 0:00 |
| b2 | Area model | `areamodel` | 0:08 |
| **b3** | **3D surfaces** | **`surface`** | **0:18** |
| **b4** | **Concrete gap** | **`surface`** | **0:30** |

**b3 and b4 are the frames to hold on**, and they are one argument in two halves.

b3 puts both rules in the same space: `(a+b)²` in green and `a²+b²` in red, drawn
as two surfaces over the same square of inputs, with the camera orbiting them. The
two sheets **touch along the two axes** and separate everywhere else, which is the
part a derivation cannot say: the student's rule is not merely wrong, it is exactly
right whenever `a` or `b` is zero, and that is why it feels right. Every case they
have ever checked it on was probably one of those.

b4 turns the picture into a number. A yellow bar rises between the two sheets at
the student's own values, and the bottom of the frame reads
`a²+b²: 10 · (a+b)²: 16 · missing: 6`. The bar is the six they dropped, at the
height it actually has.

The area model in b2 stays because the two beats do different jobs: the square
**counts** the missing terms, the surfaces **measure** them. Neither replaces the
other and the video is stronger for having both.

## The golden video is narrated

One voice throughout, four lines, on `s2.1-pro-free`. **Play it with sound on.**
The untouched render is kept beside it as `silent.mp4`.

Read end to end, which is how it was written:

> You thought squaring a bracket means squaring each term, but it doesn't. That's
> why the middle term appears: two rectangles in the square of side a plus b make
> two a b. So the missing term changes the surface shape. At y equals one, your
> value is ten, the correct is sixteen, and the six missing is exactly that middle
> term.

Two properties to listen for, both of which the first version of this got wrong.

**It never states the student's rule without denying it in the same breath.** The
earlier script opened *"You thought a plus b, all squared, equals a squared plus b
squared."* and then moved on to numbers. Said out loud with nothing after it, that
sentence reads the student's own mistake back to them as fact, and the video has
taught them the error. Now the contradiction is attached: *"but that's the mistake;
it leaves out the middle term."*

**Every line reaches back to the one before it.** "That's why", "So", "At". The
animation cuts between sections and the voice is the only thing carrying the
student across the cut; without those joins, four beats sound like four separate
videos.

It also **names what is on screen** rather than restating the algebra: "two
rectangles in the square of side a plus b" over the beat that draws exactly those
rectangles, and "the six missing" over the beat that draws the bar.

## The video fits the explanation, not the reverse

Until recently the script was the thing that gave way: a beat's word budget came
from its measured duration, so a 6-second beat got sixteen words, and sixteen words
cannot state a rule, contradict it, and give the reason. The script wrote the label
instead. The animation was not under-narrated — it was too short to be explained
over. The chain-rule session below carries 114 spoken words, which is around
45 seconds of speech, against a 40s render.

So each beat is now **held open on its own last frame** for exactly as long as its
line needs. The picture is untouched and nothing is stretched or sped up; the video
grows only at beat boundaries, where nothing is moving.

| Session | Render | Published | Held |
|---|---|---|---|
| `(y+3)^2` | 35.6s | 38.0s | +2.4s |
| `d/dx sin(x^2)` | 40.0s | 44.9s | +4.9s |

On the gap-pillar beat that means the bar and its three readings stay on screen
while the voice finishes explaining them, instead of cutting away the moment the
bar finished drawing.

Which is also why beat timings are rewritten after narration. The published video
is no longer the render, and a citation that seeks to `[beat:b3]` has to land on b3
in the file the page actually serves. `silent.mp4` and `silent.spans.json` beside it
keep the render and its own timings, so re-narrating computes the same answer from
the same input rather than padding its own output.

Every variable is spoken with forced phonemes, so "y" is the letter and not "ee",
and "a" is the letter and not the article.

The voice is pinned to `ba1cd26ca87b42b2bf7d60c1f65f9242` ("Adam - Calm, Smart").
That is not cosmetic: every beat is a separate API request, so an unset voice
gives one video a different narrator on every beat.

To change voice, set `FISH_VOICE_ID` and then re-measure the rate, because the
word budget is calibrated to the voice:

```bash
uv run python scripts/measure_voice.py <voice-id>
```

Candidates measured on the same five lines, for reference:

| Voice | Mean w/s | Slowest | Spread |
|---|---|---|---|
| Adam - Calm, Smart | 2.84 | 2.57 | 0.74 |
| calm storyteller male | 2.73 | 2.50 | 0.66 |
| Jon - Warm & Grounded | 2.33 | 2.04 | 0.65 |
| Nathan - Audiobook | 2.21 | 2.04 | 0.59 |
| CALM- NORMAL | 2.14 | 1.80 | 0.73 |

Every voice slows on dense spoken maths, so the budget is set just under the
*slowest* line rather than the mean.

To re-narrate after any change, which costs one model call and a few seconds of
TTS against a render that already exists:

```bash
uv run python scripts/narrate_session.py fa8f9e92-220d-4345-b9e5-484a228fb6ff
```

Safe to run repeatedly: it reads `silent.mp4` and republishes over `video.mp4`,
so the URL the page already serves keeps working.

## Before recording

Adopt the golden session's anonymous handle so **Insights → Your patterns**
shows history rather than an empty state. Paste into the browser console once:

```js
localStorage.setItem("astray.handle", "astray-final"); location.reload()
```

The handle is a random per-browser id with no other meaning; the page never
displays it and `/api/insights` never returns it.

## The run of show

1. **Submit** — `/#/`. Type a problem and working to show the input, or run a
   live diagnosis if you want one on camera (~20s to the diagnosis card, the
   animation keeps building behind it).
2. **Diagnosis card** — falsifiable rule `(a+b)^2 -> a^2 + b^2`, the plain
   statement, and four badges: *✓ checked with SymPy · diverges at step 1 ·
   confidence 95% · 2 other students made this error*. The SymPy badge is the
   measured result, not the model's claim.
3. **The animation and the rail** — click **0:18 3D surfaces**; the player seeks to
   that beat's measured start, and the rail's active chip follows the playhead as
   the video runs. Every chip is also a real accessible control ("Jump to 0:18, 3D
   surfaces"), so the rail is keyboard-reachable.
4. **Chat** — say *"Hey Astray, why doesn't my rule work"* (see *Asking by voice*
   below; grant the permission and reload **before** recording, and press the
   microphone button instead if the phrase misses). Three exchanges are already in
   the database and every reply carries
   chips. The one to read aloud is *"Is my answer ever right, or is it always
   wrong?"*: the tutor answers the actual question, cites four beats, and describes
   the area model by its parts ("split into four pieces, `y^2`, `3y`, `3y`, and
   `9`"). It can talk about the geometry because the geometry is there.
5. **Insights** — `/#/insights`. *Your patterns* lists three different
   misconceptions, one per demo session: *freshman's dream*,
   *square-root-single-sign*, *chain-rule-omitted*. *Across everyone* shows
   **3 students** on the binomial one, and the diagnosis card said *2 others* for
   the same reason: one of those three wrote `(x + 5)^2`, a different problem in
   different letters that canonicalised onto the same entry. The canonical
   statements are the taxonomy's own words, not the model's phrasing for this
   session, which is what makes them countable.

## Asking by voice: "Hey Astray"

Say **"Hey Astray, why doesn't my rule work"** and it answers, out loud. No button.

### It decides whether you changed the subject

"Hey Astray" is not only a way to ask about the animation on screen. Every spoken
sentence is routed first, and the routing is the point: a follow-up goes to this
conversation, a *different problem* becomes a new problem with its own animation.

> "Hey Astray, I still don't get the binomial thing" → answered here.
>
> "Hey Astray, I was working on differential equations and I couldn't figure out
> dy/dx equals 2x minus 5 over y squared" → a new session, diagnosed and animated.

The second one is the case this was built for, because it used to fail silently.
Said mid-session, it was answered as a question about expanding a bracket: a reply
grounded in the wrong animation, and no animation ever built for what was asked.

The same call repairs the speech, which is not optional. Chrome delivered that
sentence as **"I was solving DUI over DX = 2x - 5 over y squared"** — verbatim,
from a real session — and `dy/dx` has to come back before anything can diagnose it.
It is speech-to-text, not a maths recogniser, so the repair is a model call, not a
regex. Ambiguous cases go to the conversation on purpose: a new problem answered
in the existing chat is one sentence to correct, where a follow-up sent to a
three-minute render is not.

What was heard is printed above the diagnosis, because a spoken problem was never
typed anywhere you can check it.

### You do not have to have tried anything

Steps are the premise of this product, not a requirement of it. "I kept getting
stuck and I don't know what to do" is a real thing to say, and there is no wrong
step in it to find.

So a session with no working shown **explains the method** instead of diagnosing a
mistake. The card reads *How to solve it* over the solved steps rather than *Here's
where it went astray*, and the animation teaches the method.

This is a framing swap, not a second pipeline: every stage from s2 to s7 reads the
diagnosis' `buggy_rule` and `misconception_statement`, so those two fields are
replaced once, in `pipeline.as_explainer`, and each stage carries on doing its job.
Narration gets a separate prompt, because that is the one place the wrong framing
does real harm: telling a student *"you thought a plus b, all squared, equals a
squared plus b squared"* when they never said any such thing invents a mistake and
then attributes it to them.

Worth knowing why this needed doing at all: the first live run of a spoken problem
with no working came back with a storyboard containing a beat titled **"Buggy
Method"**. Every stage had been told to target a misconception, so lacking one, it
made one up.

### Everything you have asked is on the front page

**Your problems**, under the form on `/#/`. Newest first, with what each one turned
out to be. This is how you get back to a problem you *spoke*: there was no form
submission behind it and no URL you typed.

The state is written out beside the chat heading, always, in words: *Hey Astray:
listening* / *your turn* / *paused* / *off*. Click that chip to mute. The
microphone button next to the composer shows the same three states and is also
the direct route -- press it and it takes a question immediately, wake phrase or
not, which is the thing to reach for if the room is loud.

**Grant the microphone before you record.** Idle listening only starts on its own
when the permission is already granted; when it isn't, the controls sit in *off*
and wait for a click, because auto-arming into a surprise permission prompt is the
behaviour that makes an always-on microphone feel like something done to you.
Open the page, click the chip once, accept, reload.

**Chrome or Edge only.** Both controls hide themselves entirely on Safari and
Firefox, which have no `SpeechRecognition` at all.

### What to expect, honestly

**The wake phrase is edit distance on a general transcript, not a keyword model.**
Chrome hears "hey a stray", "hey astro", "hey ashtray"; all of those match, by
design. It will still miss occasionally and fire occasionally when it shouldn't.
A real wake word (Porcupine, openWakeWord) is a trained on-device model, which is
a different dependency and a trained keyword file. **If it misses on camera, press
the microphone button** -- same turn, same everything, no wake phrase needed.

**Idle listening degrades in a background tab.** Chrome throttles timers in a
hidden tab, and always-on here is a restart loop built on a timer, so the gap
between sessions stretches from a fifth of a second to a second or more, and
after a few minutes hidden it may stop restarting altogether. Keep the tab
fronted. (Found while testing: a stub run that should have taken two seconds took
twenty, purely from throttling.)

**There is no always-on API.** `continuous` is a request. Chrome ends a session by
itself after roughly a minute, and sooner on silence, and restarting from `onend`
*is* the mechanism. A recogniser that ends instantly five times in a row is
treated as broken and listening stops rather than spinning.

**The recogniser streams audio to Google's servers.** Idle listening means an open
connection carrying whatever is said near the machine. That is the cost of
always-on on this API, and it is why muting is a labelled control in the chat
header rather than a preference hidden somewhere.

**It is general speech-to-text, not a maths recogniser.** It hears "y squared plus
nine" and it will not reliably hear "(y+3)²". **Type the expression, speak the
question.**

### The feedback loop, and why the video pauses

While the animation is playing, the microphone is **closed**, not merely ignored:
the state chip reads *paused*. The narration is a voice explaining the student's
own mistake, in sentences full of the same words they would ask about -- and the
tagline it reads aloud contains the wake word's own root. A live microphone hears
the tutor, transcribes it, and either wakes on it or files the explanation as the
student's next question. Muting the element is not enough on a laptop with open
speakers, so the recogniser is shut. Listening resumes about eight tenths of a
second after the audio stops, which covers the room's echo of it.

Landing on the wake phrase pauses the video for the same reason, and plays a short
blip so you know to start talking.

## The other two, each arguing a different way

| Session | Problem | The beat that argues |
|---|---|---|
| `bb6a4531-dbde-4d4e-953c-6237dd1e3245` | `d/dx sin(x²)` → `cos(x²)` | `lift`: the composition in space, then equal steps in `x` becoming unequal steps in `u` |
| `5901c1da-fe6b-4e55-a4f6-31bd5834cb38` | `x^2 = 16` → `x = 4` | `numberline`: a second dot lands at −4 where the student had nothing |

**The chain rule is the second cinematic one**, 44.9s, five beats, and it is the
clearest case for giving the middle quantity its own axis. b3 draws `sin(x²)` as a
single curve in space whose three shadows are the three stages: `u = x²` on the
floor, `sin(u)` up the wall, and the answer on the back. The middle value is
invisible in flat algebra, which is exactly why the factor that comes from it goes
missing.

b4 is the one that names the error. Equal steps along `x` are carried up to the
inner curve and across, and the marks they leave are bunched near zero and spread
apart at the ends. That spacing ratio *is* `2x`. The student did not write a wrong
number; they wrote an answer that assumes the middle quantity keeps pace with `x`,
and the picture is that assumption failing.

**The lost-root case is deliberately left flat**, and it is the clearest argument
for having more than one primitive rather than one impressive one. Every line the
student wrote is true, so there is nothing to cross out and a side-by-side
comparison has no content; the error is an *absence*, and on a line an absence is a
place. A surface would add nothing here but spectacle.

## Worth knowing before you record

**Per-run variation is real, and it has one cause.** Beats built from primitives
render cleanly; beats where the model positions its own mobjects sometimes do not.
Across a dozen runs the failures were all the same shape: a diagram at a quarter of
its intended size jammed against the frame edge, or three lines of text stacked
past the bottom. `layout.fit` and the `caption` stacking exist for exactly this and
are now in the s7 prompt, but a hand-built beat can still come out worse than a
primitive one. If a fresh render looks wrong, that is where to look first.

## Do not use this session on camera

`e4c561a6-402e-442d-ad2a-e6f5508b6dc5` (`(x + 5)^2`) is a peer that makes the
insights count real, and it has chat history, but it was rendered before the
primitives existed and its comparison beat is unreadable: the student's own
expression is set smaller than the correct one with a thick red **X** drawn
straight over it. It exists to be counted, not to be shown.

`12a959d5-3a99-4343-a360-b681dd2aebbc` is the same `(y+3)^2` problem from before
the primitives, and it is worth opening *deliberately* beside the golden case: six
beats of white maths text on black, 34.8s, three render attempts and two repair
rounds. Same pipeline, same problem, no picture.

`84b52e28-05dd-4a33-a295-e3a3b6c167e7` is the same problem again, from after the
flat primitives and before the spatial ones: five beats, 43.6s, a correct and
readable area model and nothing in three dimensions. Open it if someone asks what
the surfaces actually added.
