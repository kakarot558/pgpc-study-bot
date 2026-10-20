"""StudyBuddy - a local, dataset-driven study chatbot (Flask).

Run from the project root:   python app.py     then open http://127.0.0.1:5000
Nothing here calls an external AI service; all matching happens on this computer.
"""
import logging
import random
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import config
from ml.feedback import FeedbackError, FeedbackStore, build_entry
from ml.predict import load_or_build_retriever
from ml.preprocess import DatasetError

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_message(payload, max_length: int):
    """Return (clean_message, error_text). Exactly one of them is None."""
    if not isinstance(payload, dict):
        return None, "Send a JSON object like {\"message\": \"your question\"}."
    message = payload.get("message")
    if not isinstance(message, str):
        return None, "The 'message' field must be text."
    message = " ".join(CONTROL_CHARS.sub("", message).split())
    if not message:
        return None, "Please type a question first."
    if len(message) > max_length:
        return None, f"Please keep your question under {max_length} characters."
    if not re.search(r"[A-Za-z0-9]", message):
        return None, "Please include some letters or numbers in your question."
    return message, None


CLOUD_SYNC = re.compile(r"onedrive|dropbox|google ?drive|icloud|box sync|sync\.com|mega ?sync", re.IGNORECASE)


def sync_folder_warning(path) -> str:
    """Warn if the feedback file lives in a cloud-synced folder, where student questions could be uploaded."""
    if not CLOUD_SYNC.search(str(path)):
        return ""
    return (f"PRIVACY WARNING: the feedback file ({path}) is inside a cloud-synced folder (OneDrive, Dropbox, "
            "Google Drive or similar). If students click Helpful / Not helpful, their question text may be uploaded "
            "to that cloud service. Set STUDYBUDDY_FEEDBACK_DIR to a local folder such as C:\\StudyBuddyData, "
            "or move the project folder out of the synced location, before running a pilot.")


def create_app(test_config: dict = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        FAQ_PATH=str(config.FAQ_PATH), RETRIEVER_ARTIFACT=str(config.RETRIEVER_ARTIFACT),
        SIMILARITY_THRESHOLD=config.SIMILARITY_THRESHOLD,
        MAX_MESSAGE_LENGTH=config.MAX_MESSAGE_LENGTH, MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
        FEEDBACK_PATH=str(config.FEEDBACK_PATH), LOG_DECLINED=config.LOG_DECLINED,
    )
    if test_config:
        app.config.update(test_config)
    logging.basicConfig(level=logging.INFO)

    app.extensions["feedback"] = FeedbackStore(app.config["FEEDBACK_PATH"])
    app.extensions["sync_warning"] = sync_folder_warning(Path(app.config["FEEDBACK_PATH"]).resolve())
    if app.extensions["sync_warning"]:
        app.logger.warning(app.extensions["sync_warning"])

    # Load the knowledge base once at start-up (never once per chat message).
    app.extensions["retriever"] = None
    app.extensions["load_error"] = None
    try:
        retriever, source = load_or_build_retriever(
            app.config["FAQ_PATH"], app.config["RETRIEVER_ARTIFACT"], app.config["SIMILARITY_THRESHOLD"])
        app.extensions["retriever"] = retriever
        app.logger.info("Knowledge base ready: %d questions (%s).", retriever.size, source)
    except (DatasetError, ValueError, OSError) as exc:
        app.extensions["load_error"] = str(exc)
        app.logger.error("Could not load the knowledge base: %s", exc)

    def retriever_or_none():
        return app.extensions["retriever"]

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        r = retriever_or_none()
        topics = [{"name": t["name"], "count": t["count"]} for t in r.topic_summary()] if r else []
        suggestions = []
        if r:
            for t in r.topic_summary():
                suggestions += [{"question": q, "topic": t["name"]} for q in t["questions"][:2]]
        return render_template("index.html", topics=topics, suggestions=suggestions, ready=r is not None,
                               max_length=app.config["MAX_MESSAGE_LENGTH"])

    @app.get("/topics")
    def topics():
        r = retriever_or_none()
        return render_template("topics.html", topics=r.topic_summary() if r else [], ready=r is not None)

    @app.get("/quiz")
    def quiz():
        r = retriever_or_none()
        subjects = [{"name": t["name"], "cards": len(r.flashcards(t["name"]))} for t in r.topic_summary()] if r else []
        return render_template("quiz.html", subjects=subjects, ready=r is not None)

    @app.get("/api/flashcard")
    def flashcard():
        """A random flashcard (one per distinct answer). Query: subject, exclude=1,2,3, seed (testing)."""
        r = retriever_or_none()
        if r is None:
            return jsonify(error="The knowledge base is not available."), 503
        subject = request.args.get("subject", "").strip() or None
        if subject and subject not in {t["name"] for t in r.topic_summary()}:
            return jsonify(error="Unknown subject."), 400
        raw = request.args.get("exclude", "")
        if len(raw) > 4000 or not re.fullmatch(r"[0-9,]*", raw):
            return jsonify(error="Invalid exclude list."), 400
        seen = {int(x) for x in raw.split(",") if x}
        cards = [c for c in r.flashcards(subject) if c["id"] not in seen]
        if not cards:
            return jsonify(done=True, remaining=0)
        seed = request.args.get("seed")
        rng = random.Random(int(seed)) if seed and seed.isdigit() else random.Random()
        card = rng.choice(cards)
        return jsonify(done=False, remaining=len(cards) - 1, **card)

    @app.get("/about")
    def about():
        r = retriever_or_none()
        return render_template("about.html", size=r.size if r else 0, threshold=app.config["SIMILARITY_THRESHOLD"],
                               reviewed=r.reviewed_count if r else 0, log_declined=app.config["LOG_DECLINED"],
                               sync_warning=bool(app.extensions["sync_warning"]),
                               subjects=[t["name"] for t in r.topic_summary()] if r else [])

    @app.get("/health")
    def health():
        r = retriever_or_none()
        if r is None:
            return jsonify(status="error", knowledge_base_loaded=False,
                           detail="Knowledge base could not be loaded. Check data/study_faq.csv."), 503
        return jsonify(status="ok", knowledge_base_loaded=True, questions=r.size,
                       topics=len(r.topic_summary()))

    # ------------------------------------------------------------------- chat
    @app.post("/chat")
    def chat():
        r = retriever_or_none()
        if r is None:
            return jsonify(error="The knowledge base is not available. Please check the server console."), 503
        message, problem = validate_message(request.get_json(silent=True), app.config["MAX_MESSAGE_LENGTH"])
        if problem:
            return jsonify(error=problem), 400
        try:
            result = r.answer(message)
        except Exception:                                  # never show a stack trace to the student
            app.logger.exception("Unexpected error while answering a question")
            return jsonify(error="Something went wrong while looking for an answer. Please try again."), 500
        if app.config["LOG_DECLINED"] and not result["match_found"]:
            try:                                        # optional, never allowed to break the chat
                app.extensions["feedback"].add(build_entry(message, "declined", None, result["similarity_score"], False))
            except Exception:
                app.logger.warning("Could not log a declined question", exc_info=True)
        return jsonify(answer=result["answer"], topic=result["topic"],
                       matched_question=result["matched_question"],
                       similarity_score=result["similarity_score"], match_found=result["match_found"],
                       suggestion=result["suggestion"], confidence=result["confidence"],
                       related_questions=result["related_questions"])

    @app.post("/feedback")
    def feedback():
        payload = request.get_json(silent=True)
        message, problem = validate_message(payload, app.config["MAX_MESSAGE_LENGTH"])
        if problem:
            return jsonify(error=problem), 400
        try:
            entry = build_entry(message, payload.get("rating"), payload.get("matched_question"),
                                payload.get("similarity_score"), payload.get("match_found"),
                                app.config["MAX_MESSAGE_LENGTH"])
            if entry["rating"] == "declined":            # students may only send up / down
                raise FeedbackError("rating must be 'up' or 'down'.")
            app.extensions["feedback"].add(entry)
        except FeedbackError as exc:
            return jsonify(error=str(exc)), 400
        except OSError:
            app.logger.exception("Could not write feedback")
            return jsonify(error="Feedback could not be saved."), 500
        return jsonify(ok=True)

    # ----------------------------------------------------------------- errors
    def wants_json():
        return request.path == "/chat"

    @app.errorhandler(404)
    def not_found(_):
        return (jsonify(error="Not found."), 404) if wants_json() else (render_template("error.html", code=404, text="That page does not exist."), 404)

    @app.errorhandler(405)
    def wrong_method(_):
        return jsonify(error="Method not allowed."), 405

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="That message is too large."), 413

    @app.errorhandler(500)
    def server_error(_):
        return jsonify(error="Unexpected server error."), 500

    return app


def check_host(host: str, debug: bool) -> str:
    """Return a warning if the app will be reachable from other computers; refuse debug mode in that case."""
    if host in ("127.0.0.1", "localhost", "::1"):
        return ""
    if debug:
        raise SystemExit("Refusing to start: debug mode must never be combined with a network-visible host "
                         "(the debugger allows remote code execution). Unset STUDYBUDDY_DEBUG.")
    return (f"WARNING: listening on {host} - other computers on your network can open this app. "
            "There is no login. Use only on a trusted network, and stop it when the pilot ends.")


app = create_app()

if __name__ == "__main__":
    # Default 127.0.0.1 = reachable only from this computer. Debug stays off unless STUDYBUDDY_DEBUG=1.
    notice = check_host(config.HOST, config.DEBUG)
    if notice:
        print(notice)
    print(f"StudyBuddy: open http://127.0.0.1:{config.PORT} in your browser (Ctrl+C once to stop). "
          "The first start can take up to a minute while Python loads its libraries.", flush=True)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
