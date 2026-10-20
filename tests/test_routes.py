import json
from pathlib import Path

import pytest

from app import create_app, validate_message


@pytest.mark.parametrize("path", ["/", "/topics", "/about"])
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200 and b"StudyBuddy" in response.data


def test_home_shows_local_notice_and_suggestions(client):
    html = client.get("/").get_data(as_text=True)
    assert "no external AI API" in html and "What is a banana?" in html and 'id="clear-btn"' in html


def test_health_ok(client):
    data = client.get("/health").get_json()
    assert data["status"] == "ok" and data["knowledge_base_loaded"] is True and data["questions"] == 4


def test_chat_known_question(client):
    response = client.post("/chat", json={"message": "What is a banana?"})
    data = response.get_json()
    assert response.status_code == 200
    assert set(data) >= {"answer", "topic", "matched_question", "similarity_score", "match_found"}
    assert data["match_found"] is True and data["answer"] == "A banana is a yellow fruit."


def test_chat_unknown_question_returns_fallback(client):
    data = client.post("/chat", json={"message": "Who painted the Mona Lisa?"}).get_json()
    assert data["match_found"] is False and data["matched_question"] is None and "could not find" in data["answer"]


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "   "}, {"message": "?!?"}, {"message": 42},
                                     {"message": None}, {"other": "x"}, [], "text"])
def test_chat_rejects_bad_payloads(client, payload):
    response = client.post("/chat", json=payload)
    assert response.status_code == 400 and "error" in response.get_json()


def test_chat_rejects_invalid_json(client):
    for body, ctype in [("{not json", "application/json"), ("hello", "text/plain"), ("", "application/json")]:
        response = client.post("/chat", data=body, content_type=ctype)
        assert response.status_code == 400 and "error" in response.get_json()


def test_chat_rejects_too_long_message(client):
    assert client.post("/chat", json={"message": "a" * 301}).status_code == 400
    assert client.post("/chat", json={"message": "a" * 5000 + " " * 20000}).status_code in (400, 413)


def test_chat_get_not_allowed(client):
    assert client.get("/chat").status_code == 405


def test_unexpected_error_hides_details(app, client, monkeypatch):
    def boom(_):
        raise RuntimeError("secret internal detail /home/user/file.py")
    monkeypatch.setattr(app.extensions["retriever"], "answer", boom)
    response = client.post("/chat", json={"message": "What is a banana?"})
    body = response.get_data(as_text=True)
    assert response.status_code == 500 and "secret" not in body and "Traceback" not in body


def test_unknown_page_is_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_security_headers(client):
    headers = client.get("/").headers
    assert "default-src 'self'" in headers["Content-Security-Policy"] and headers["X-Content-Type-Options"] == "nosniff"


def test_app_survives_missing_dataset(tmp_path):
    broken = create_app({"TESTING": True, "FAQ_PATH": str(tmp_path / "missing.csv"),
                         "RETRIEVER_ARTIFACT": str(tmp_path / "none.joblib")})
    c = broken.test_client()
    assert c.get("/health").status_code == 503
    assert c.post("/chat", json={"message": "hi there"}).status_code == 503
    assert c.get("/").status_code == 200 and c.get("/topics").status_code == 200


def test_validate_message_cleans_input():
    msg, err = validate_message({"message": "  hello\x00   world \n"}, 50)
    assert msg == "hello world" and err is None


def test_templates_have_no_external_urls_or_inline_scripts():
    root = Path(__file__).resolve().parent.parent
    for path in list((root / "templates").glob("*.html")) + [root / "static/js/chat.js", root / "static/css/style.css"]:
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, path.name
    for path in (root / "templates").glob("*.html"):
        assert "<script>" not in path.read_text(encoding="utf-8")


def test_frontend_uses_safe_dom_methods():
    js = (Path(__file__).resolve().parent.parent / "static/js/chat.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "textContent" in js and "outerHTML" not in js
