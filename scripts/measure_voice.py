"""Measure a Fish Audio voice's speaking rate, so the word budget is calibrated.

`Settings.narration_words_per_second` decides how many words a beat can hold, and
getting it wrong is the difference between narration that fits and narration that
talks over the next visual. The right value is a property of the chosen voice, not
a constant: across candidate voices the same two sentences ranged from 7.7s to
10.5s, a 36% spread.

This synthesises a fixed set of lines, deliberately varied between mostly prose
and mostly spoken maths, and reports words per second for each. The spread matters
as much as the mean: a voice with a wide spread cannot be budgeted for tightly,
whatever its average, so the run reports both and suggests a rate at the slow end.

    uv run python scripts/measure_voice.py                    # the configured voice
    uv run python scripts/measure_voice.py <voice-id> [...]   # compare candidates
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.audio.fish import FishAudioClient  # noqa: E402
from server.audio.speech import with_letter_phonemes, word_count  # noqa: E402
from server.config import get_settings  # noqa: E402

# Deliberately spans the range the narrator actually has to cover: the first two
# are ordinary prose, the last three are dense spoken maths, which is consistently
# the slower end.
LINES = [
    "Squaring a bracket isn't the same as squaring each piece separately.",
    "Those two middle terms are what you dropped, and together they make six y.",
    "y plus three, all squared, just means y plus three times itself.",
    "So you get y squared, then three y twice over, then nine.",
    "Try y equals one. The real answer is sixteen, and yours gives ten.",
]


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


async def measure(voice_id: str | None, label: str) -> dict:
    settings = get_settings()
    client = FishAudioClient(
        settings.fish_api_key or "",
        base_url=settings.fish_base_url,
        model=settings.fish_model,
        voice_id=voice_id,
        speed=settings.narration_speed,
        timeout_s=settings.narration_timeout_s,
    )
    rates, total_words, total_s = [], 0, 0.0
    try:
        # Measure what is actually sent, phoneme tags included. Forcing the letter
        # names costs about 5% in duration on maths-heavy lines, and a rate
        # calibrated on untagged text is optimistic by exactly that much.
        clips = await asyncio.gather(
            *(client.synthesize(with_letter_phonemes(line)) for line in LINES),
            return_exceptions=True,
        )
    finally:
        await client.aclose()

    with tempfile.TemporaryDirectory() as tmp:
        for index, (line, clip) in enumerate(zip(LINES, clips, strict=True)):
            if isinstance(clip, BaseException):
                print(f"  line {index + 1}: FAILED {type(clip).__name__}: {str(clip)[:80]}")
                continue
            path = Path(tmp) / f"{index}.mp3"
            path.write_bytes(clip.audio)
            seconds = _duration(path)
            words = word_count(line)
            rates.append(words / seconds)
            total_words += words
            total_s += seconds
            print(f"  {words:>2} words  {seconds:5.2f}s  {words / seconds:4.2f} w/s   {line[:52]}")

    if not rates:
        return {"label": label, "voice_id": voice_id, "ok": False}

    return {
        "label": label,
        "voice_id": voice_id,
        "ok": True,
        "mean": total_words / total_s,
        "slowest": min(rates),
        "fastest": max(rates),
        "spread": max(rates) - min(rates),
        "stdev": statistics.pstdev(rates),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("voice_ids", nargs="*", help="voice ids to compare; default is configured")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.fish_api_key:
        print("set FISH_API_KEY in server/.env")
        return 1

    targets = args.voice_ids or [settings.fish_voice_id]
    results = []
    for voice_id in targets:
        label = voice_id or "(configured default)"
        print(f"\n=== {label} ===")
        results.append(await measure(voice_id, label))

    good = [r for r in results if r["ok"]]
    if not good:
        return 1

    print(f"\n{'voice':<36}{'mean':>7}{'slowest':>9}{'spread':>8}{'suggest':>9}")
    for r in sorted(good, key=lambda r: r["spread"]):
        # Budget at the slow end, not the mean: a line that overruns pushes the
        # next one late, while a short line costs nothing.
        suggested = round(r["slowest"] * 0.97, 2)
        print(
            f"{r['label'][:34]:<36}{r['mean']:>7.2f}{r['slowest']:>9.2f}"
            f"{r['spread']:>8.2f}{suggested:>9.2f}"
        )

    best = min(good, key=lambda r: r["spread"])
    print(
        f"\nmost consistent pace: {best['label']} "
        f"(spread {best['spread']:.2f} w/s, stdev {best['stdev']:.2f})"
    )
    print(f"set NARRATION_WORDS_PER_SECOND={round(best['slowest'] * 0.97, 2)} for it")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
