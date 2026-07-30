"""Checks on the static frontend.

There is no JavaScript test runner in this project, so nothing here executes
`app.js` or `voice.js` -- the dictation state machine is driven against a stubbed
`SpeechRecognition` in a real browser instead, because the bugs worth catching
there are about the browser's own event ordering and a stub in Node would only
re-assert my own assumptions about it.

What these tests do cover is the wiring between the files, which is where a
silent break is most likely: a renamed id, a script that stops being served, an
asset referenced but absent. All of those fail as "the feature quietly does
nothing" in a browser and as an obvious failure here.
"""

from pathlib import Path

WEB = Path(__file__).parent.parent / "web"


def read(name: str) -> str:
    return (WEB / name).read_text()


def test_every_referenced_asset_exists():
    """A 404 on a script or a mask image is invisible until the feature is used."""
    html = read("index.html")
    css = read("style.css")
    referenced = {"voice.js", "app.js", "style.css", "astray-mark.svg", "astray-wordmark.svg"}
    referenced |= {"astray-word.svg"} if "astray-word.svg" in css else set()
    for name in referenced:
        assert name in html or name in css, f"{name} is not referenced anywhere"
        assert (WEB / name).exists(), f"{name} is referenced but missing from web/"


def test_voice_loads_before_the_app_that_uses_it():
    """Both are `defer`, which runs them in document order, so order is the
    contract: `createDictation` has to exist by the time `app.js` runs."""
    html = read("index.html")
    assert html.index("voice.js") < html.index("app.js")


def test_the_composer_ids_the_voice_wiring_expects_are_present():
    html = read("index.html")
    for needed in ('id="mic-btn"', 'id="chat-input"', 'id="chat-hint"', 'id="chat-form"'):
        assert needed in html, f"{needed} is gone; the voice wiring reads it by id"


def test_the_mic_starts_hidden_and_disabled():
    """It is revealed only once the browser is known to support dictation, and
    enabled only once chat opens. Shipping it visible would offer a control that
    does nothing on Safari and Firefox."""
    html = read("index.html")
    button = html[html.index('id="mic-btn"') : html.index('id="mic-btn"') + 220]
    assert "hidden" in button
    assert "disabled" in button


def test_the_hidden_attribute_beats_the_button_display_rule():
    """`hidden` is only a UA-stylesheet `display: none`, and `.btn` sets
    `display: inline-flex`, which outranks it. Without an author rule the
    microphone renders on exactly the browsers that cannot use it."""
    assert "[hidden] { display: none !important; }" in read("style.css")


def test_the_send_button_is_selected_by_type():
    """`#chat-form button` used to be unambiguous. The mic button is now first in
    the form, so a bare selector enables the microphone and leaves send dead."""
    assert "#chat-form button[type=submit]" in read("app.js")


def test_no_all_caps_text_transform():
    """Sentence case is the house style. `text-transform: uppercase` shouts, and
    it also hides the real casing of the string from anyone reading the markup."""
    css = read("style.css")
    assert "uppercase" not in css


def test_the_cursor_spotlight_sits_behind_card_content():
    """Without the negative z-index this layer paints over the inputs: `screen`
    spares near-white text but washes coral across every field's dark surface."""
    css = read("style.css")
    spotlight = css[css.index(".card-lit::before") : css.index(".card-lit:hover::before")]
    assert "z-index: -1;" in spotlight


def test_the_wordmark_is_masked_rather_than_an_image():
    """The lettering is filled `currentColor`, which resolves to black inside an
    <img>'s own document -- invisible on this page. A mask inverts the
    relationship so the page's colour drives it."""
    css = read("style.css")
    wordmark = css[css.index(".wordmark {") : css.index(".diagnosis > p")]
    assert "mask:" in wordmark
    assert "background-color: currentColor" in wordmark
    assert "astray-word.svg" in wordmark
