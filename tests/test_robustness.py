"""Fuzz, concurrency and abuse tests for the web routes (all offline)."""
import random
import string
from concurrent.futures import ThreadPoolExecutor

import pytest

NASTY = ["'; DROP TABLE users; --", "__import__('os').system('echo hacked')", "{{7*7}}", "${7*7}", "<script>alert(1)</script>",
         "../../etc/passwd", "%s%s%s%n", "\u202eevil\u202c text", "😀😀😀 what is a banana 😀", "\x00\x01 banana \x1f",
         "A" * 299, "banana\n\r\n\tbanana", "ＷＨＡＴ ＩＳ Ａ ＢＡＮＡＮＡ", "null", "NaN", "True", "-1"]


def random_text(rng, n):
    pool = string.printable + "éüñ日本語العربيةЖ😀\u200b\u202e"
    return "".join(rng.choice(pool) for _ in range(n))


@pytest.mark.parametrize("text", NASTY)
def test_nasty_inputs_never_crash(client, text):
    response = client.post("/chat", json={"message": text})
    assert response.status_code in (200, 400)
    body = response.get_json()
    assert body is not None
    if response.status_code == 200:
        assert "49" not in body["answer"] and "hacked" not in body["answer"]
        assert 0.0 <= body["similarity_score"] <= 1.0


def test_random_fuzzing_never_returns_500(client):
    rng = random.Random(1234)
    for _ in range(300):
        text = random_text(rng, rng.randint(0, 340))
        response = client.post("/chat", json={"message": text})
        assert response.status_code in (200, 400), repr(text)
        assert response.get_json() is not None


@pytest.mark.parametrize("payload", [{"message": ["banana"]}, {"message": {"a": 1}}, {"message": True}, {"message": 1.5},
                                     {"message": "banana", "extra": {"deep": [1] * 100}}, None, 5, "banana"])
def test_odd_json_shapes(client, payload):
    response = client.post("/chat", json=payload)
    assert response.status_code in (200, 400)


def test_wrong_content_types_and_methods(client):
    assert client.post("/chat", data="message=banana", content_type="application/x-www-form-urlencoded").status_code == 400
    for method in ("put", "delete", "patch"):
        assert getattr(client, method)("/chat", json={"message": "banana"}).status_code == 405
    assert client.post("/health").status_code == 405


def test_huge_body_rejected(app):
    client = app.test_client()
    response = client.post("/chat", data='{"message": "' + "a" * 50000 + '"}', content_type="application/json")
    assert response.status_code == 413


def test_same_question_same_answer(client):
    first = client.post("/chat", json={"message": "What is a rocket?"}).get_json()
    for _ in range(20):
        assert client.post("/chat", json={"message": "What is a rocket?"}).get_json() == first


def test_concurrent_requests_are_consistent(app):
    questions = ["What is a banana?", "what is a rocket", "how do I bake bread", "Pomodoro technique", "who painted the mona lisa"]
    expected = {q: app.test_client().post("/chat", json={"message": q}).get_json() for q in questions}

    def worker(i):
        q = questions[i % len(questions)]
        response = app.test_client().post("/chat", json={"message": q})
        return q, response.status_code, response.get_json()

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(worker, range(400)))
    assert all(status == 200 and body == expected[q] for q, status, body in results)


def test_every_page_has_a_title_and_lang(client):
    for path in ["/", "/topics", "/about"]:
        html = client.get(path).get_data(as_text=True)
        assert '<html lang="en">' in html and "<title>" in html and 'name="viewport"' in html


def test_form_controls_are_labelled(client):
    html = client.get("/").get_data(as_text=True)
    assert 'for="chat-input"' in html and 'id="chat-input"' in html and 'aria-live="polite"' in html
