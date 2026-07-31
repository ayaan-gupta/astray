#!/usr/bin/env bash
# Cut the README's animations out of real rendered sessions.
#
# The GIFs in docs/assets are not mockups and not hand-made: each one is a
# window onto a beat of a video this pipeline actually produced, cut at the beat
# boundaries the renderer reported. Regenerating them is how you keep the README
# honest after the builders change -- re-render the session, re-run this.
#
# Two-pass palette is not optional. A single pass gives GIF's default 216-colour
# web palette, which posterises a shaded surface into bands and makes the two
# sheets in the binomial scene look like contour lines rather than surfaces.
#
#   scripts/make_readme_gifs.sh
#
# Requires ffmpeg on the host (brew install ffmpeg). Reads from media/, which is
# gitignored; writes to docs/assets/, which is not.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=docs/assets
mkdir -p "$OUT"

# session-id-prefix  start  duration  width  name
#
# Each window sits *inside* one beat and starts late enough that its first frame
# already carries the argument. That offset is the whole trick, and getting it
# wrong is invisible until you look: a window opening on the beat boundary
# catches the scene mid-construction, so the binomial clip began on a lone red
# sheet with nothing to compare it to, and the lost-root clip began on a
# numberline whose second solution had not landed yet. Both read as a bug.
#
# Widths differ on purpose: the two spatial scenes are the ones worth the bytes.
CLIPS=(
  "fa8f9e92 21.5 8.0 640 surfaces"   # both sheets up, camera orbiting
  "bb6a4531 28.3 7.7 640 lift"       # pace marks landing on the built composition
  "5901c1da 20.8 3.0 520 numberline" # the second root arriving, before b3 ends at 23.9
)

for clip in "${CLIPS[@]}"; do
  read -r sid start dur width name <<<"$clip"
  src=$(find "media/$sid"* -path "*480p15/video.mp4" | head -1)
  if [ -z "$src" ]; then
    echo "skip $name: no render for $sid" >&2
    continue
  fi
  palette="${TMPDIR:-/tmp}/astray-$name-palette.png"

  ffmpeg -v error -y -ss "$start" -t "$dur" -i "$src" \
    -vf "fps=12,scale=$width:-1:flags=lanczos,palettegen=stats_mode=diff" "$palette"
  ffmpeg -v error -y -ss "$start" -t "$dur" -i "$src" -i "$palette" \
    -lavfi "fps=12,scale=$width:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
    "$OUT/$name.gif"

  printf '%-12s %s\n' "$name" "$(du -h "$OUT/$name.gif" | cut -f1)"
done
