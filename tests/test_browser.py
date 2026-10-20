"""Real-browser tests (Chromium via Playwright). Skipped automatically if Playwright/Chromium is missing.

Setup (optional):  python -m pip install playwright  &&  python -m playwright install chromium
"""
import threading

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright  # noqa: E402
from werkzeug.serving import make_server        # noqa: E402


@pytest.fixture()
def base_url(app):
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture()
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:                      # browser binary not installed
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


def ask(page, text):
    page.fill("#chat-input", text)
    page.press("#chat-input", "Enter")
    # (string-based wait_for_function would be blocked by the app's strict Content-Security-Policy)
    expect(page.locator(".typing")).to_have_count(0)
    expect(page.locator("#send-btn")).to_be_enabled()


def test_chat_flow_clear_and_keyboard(browser, base_url):
    page = browser.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(base_url + "/")
    ask(page, "What is a banana?")                                     # Enter key submits
    assert "yellow fruit" in page.locator(".bubble.bot").last.inner_text()
    assert page.locator(".bubble.user").count() == 1
    ask(page, "Who painted the Mona Lisa?")
    assert "could not find" in page.locator(".bubble.bot").last.inner_text()
    page.click("#clear-btn")
    assert page.locator(".bubble").count() == 1
    assert errors == []


def test_user_text_is_never_rendered_as_html(browser, base_url):
    page = browser.new_page()
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    page.goto(base_url + "/")
    ask(page, "<img src=x onerror=alert(1)> <b>bold</b> banana")
    assert page.locator("#messages img, #messages b").count() == 0
    assert "<b>bold</b>" in page.locator(".bubble.user").last.inner_text()
    assert dialogs == []


def test_empty_and_overlong_input(browser, base_url):
    page = browser.new_page()
    page.goto(base_url + "/")
    page.click("#send-btn")                                            # empty: browser 'required' blocks it
    assert page.locator(".bubble.user").count() == 0
    assert page.evaluate("document.getElementById('chat-input').maxLength") == 300


def test_suggestions_and_topic_filter(browser, base_url):
    page = browser.new_page()
    page.goto(base_url + "/")
    total = page.locator("#suggestions li:visible").count()
    page.click("#topic-filters .chip[data-topic='Space']")
    assert 0 < page.locator("#suggestions li:visible").count() < total
    assert page.get_attribute("#topic-filters .chip[data-topic='Space']", "aria-pressed") == "true"
    page.click("#suggestions li:visible button")
    page.wait_for_selector(".bubble.bot .meta")
    assert "Topic: Space" in page.locator(".bubble.bot").last.inner_text()


def test_related_question_buttons_work(browser, base_url):
    page = browser.new_page()
    page.goto(base_url + "/")
    ask(page, "banana rocket")
    if page.locator(".related-btn").count():
        before = page.locator(".bubble.user").count()
        page.locator(".related-btn").first.click()
        expect(page.locator(".bubble.user")).to_have_count(before + 1)


def test_topics_link_prefills_chat_box(browser, base_url):
    page = browser.new_page()
    page.goto(base_url + "/topics")
    page.locator(".topic summary").first.click()
    page.locator(".topic a").first.click()
    page.wait_for_selector("#chat-input")
    assert page.input_value("#chat-input").endswith("?")


@pytest.mark.parametrize("width", [320, 390, 768, 1280])
def test_no_horizontal_scroll_at_common_widths(browser, base_url, width):
    page = browser.new_page(viewport={"width": width, "height": 800})
    for path in ["/", "/topics", "/about"]:
        page.goto(base_url + path)
        assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth"), (path, width)


def test_server_down_shows_friendly_error(browser, base_url):
    page = browser.new_page()
    page.goto(base_url + "/")
    page.route("**/chat", lambda route: route.abort())
    ask(page, "What is a banana?")
    assert "could not reach the local server" in page.locator(".bubble.error").last.inner_text()
    page.unroute("**/chat")
    page.route("**/chat", lambda route: route.fulfill(status=500, content_type="application/json", body='{"error":"Boom"}'))
    ask(page, "What is a rocket?")
    assert "Boom" in page.locator(".bubble.error").last.inner_text()


def test_feedback_buttons_save_and_confirm(browser, base_url, app):
    page = browser.new_page()
    page.goto(base_url + "/")
    ask(page, "What is a banana?")
    page.locator(".feedback-btn", has_text="Helpful").first.click()
    expect(page.locator(".bubble.bot").last.locator(".feedback")).to_contain_text("Thanks")
    entries = app.extensions["feedback"].read_all()
    assert len(entries) == 1 and entries[0]["rating"] == "up" and entries[0]["matched_question"] == "What is a banana?"


def test_quiz_page_flow(browser, base_url):
    page = browser.new_page()
    page.goto(base_url + "/quiz")
    page.click("#quiz-start")
    expect(page.locator("#quiz-reveal")).to_be_enabled()
    assert page.locator("#quiz-answer").is_hidden()
    page.click("#quiz-reveal")
    assert page.locator("#quiz-answer").is_visible()
    page.click("#quiz-knew")
    expect(page.locator("#quiz-score")).to_contain_text("1 of 1")
    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
