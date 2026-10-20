"""Feedback loop, review workflow, phrasing/coverage/test-file tools, regression check, quiz."""
import json
import threading

import pandas as pd
import pytest

import config
from app import create_app
from ml import add_phrasings as ap
from ml import coverage as cov
from ml import feedback as fb
from ml import make_test_file as mt
from ml import regression_check as rc
from ml import review as rv
from ml.preprocess import DatasetError, normalize_text


@pytest.fixture()
def fb_app(tiny_faq_path, tmp_path):
    return create_app({"TESTING": True, "FAQ_PATH": str(tiny_faq_path), "RETRIEVER_ARTIFACT": str(tmp_path / "none.joblib"),
                       "SIMILARITY_THRESHOLD": 0.2, "FEEDBACK_PATH": str(tmp_path / "fb.jsonl")})


# ------------------------------------------------------------------ feedback
def test_build_entry_validates():
    e = fb.build_entry("  What is\n a rocket? ", "up", "What is a rocket?", 0.8, True)
    assert e["message"] == "What is a rocket?" and e["rating"] == "up" and e["time"].endswith("Z")
    for bad in [dict(message="", rating="up"), dict(message="x y", rating="maybe"), dict(message=5, rating="up"),
                dict(message="a" * 400, rating="up"), dict(message="hi", rating="up", matched_question=7),
                dict(message="hi", rating="up", similarity="high"), dict(message="hi", rating="up", similarity=True),
                dict(message="hi", rating="up", match_found="yes")]:
        with pytest.raises(fb.FeedbackError):
            fb.build_entry(**bad)


def test_store_roundtrip_corrupt_lines_and_size_limit(tmp_path):
    store = fb.FeedbackStore(tmp_path / "f.jsonl", max_bytes=600)
    store.add(fb.build_entry("what is x", "up"))
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write("{broken json\n[1,2]\n")
    assert len(store.read_all()) == 1
    with pytest.raises(fb.FeedbackError, match="full"):
        for _ in range(20):
            store.add(fb.build_entry("another question here", "down"))


def test_store_is_thread_safe(tmp_path):
    store = fb.FeedbackStore(tmp_path / "t.jsonl")
    threads = [threading.Thread(target=lambda i=i: store.add(fb.build_entry(f"question {i}", "up"))) for i in range(40)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(store.read_all()) == 40


def test_feedback_route_saves_only_valid_ratings(fb_app, tmp_path):
    c = fb_app.test_client()
    ok = c.post("/feedback", json={"message": "What is a rocket?", "rating": "down", "matched_question": "What is a rocket?",
                                   "similarity_score": 0.6, "match_found": True})
    assert ok.status_code == 200 and ok.get_json() == {"ok": True}
    for payload in [{"message": "x y", "rating": "declined"}, {"message": "", "rating": "up"}, {"message": "hello", "rating": 3},
                    {"rating": "up"}, [], {"message": "hello", "rating": "up", "similarity_score": 99}]:
        assert c.post("/feedback", json=payload).status_code == 400, payload
    assert c.post("/feedback", data="{bad", content_type="application/json").status_code == 400
    assert c.get("/feedback").status_code == 405
    entries = fb.FeedbackStore(tmp_path / "fb.jsonl").read_all()
    assert len(entries) == 1 and entries[0]["rating"] == "down"
    assert set(entries[0]) == {"time", "rating", "message", "matched_question", "similarity", "match_found"}   # no personal data


def test_nothing_is_logged_by_default_but_declined_logging_is_opt_in(tiny_faq_path, tmp_path):
    path = tmp_path / "log.jsonl"
    def make(flag):
        return create_app({"TESTING": True, "FAQ_PATH": str(tiny_faq_path), "RETRIEVER_ARTIFACT": str(tmp_path / "n.joblib"),
                           "SIMILARITY_THRESHOLD": 0.2, "FEEDBACK_PATH": str(path), "LOG_DECLINED": flag}).test_client()
    off = make(False)
    off.post("/chat", json={"message": "who painted the mona lisa"}); off.post("/chat", json={"message": "what is a banana"})
    assert not path.exists()
    on = make(True)
    on.post("/chat", json={"message": "who painted the mona lisa"}); on.post("/chat", json={"message": "what is a banana"})
    entries = fb.FeedbackStore(path).read_all()
    assert [e["rating"] for e in entries] == ["declined"] and entries[0]["message"] == "who painted the mona lisa"


def test_logging_failure_never_breaks_chat(tiny_faq_path, tmp_path):
    blocker = tmp_path / "blocker"; blocker.write_text("a file, not a folder")
    c = create_app({"TESTING": True, "FAQ_PATH": str(tiny_faq_path), "RETRIEVER_ARTIFACT": str(tmp_path / "n.joblib"),
                    "FEEDBACK_PATH": str(blocker / "x.jsonl"), "LOG_DECLINED": True}).test_client()
    assert c.post("/chat", json={"message": "who painted the mona lisa"}).status_code == 200
    assert c.post("/feedback", json={"message": "hello there", "rating": "up"}).status_code == 500


def test_summary_and_export(tmp_path):
    entries = [fb.build_entry("What is x", "down", "What is y?", 0.4, True), fb.build_entry("What is x", "down", "What is y?", 0.4, True),
               fb.build_entry("how to bake", "declined", None, 0.0, False), fb.build_entry("How to bake?", "declined", None, 0.0, False),
               fb.build_entry("fine", "up", "A?", 0.9, True)]
    s = fb.summarize(entries)
    assert (s["total"], s["up"], s["down"], s["declined"]) == (5, 1, 2, 2)
    assert s["answers_marked_unhelpful"] == [("What is y?", 2)] and s["unanswered_questions"][0][1] == 2
    out = tmp_path / "needs.csv"
    assert fb.export_needs_label(s, out) == 2           # the declined question AND the 'Not helpful' wrong match
    exported = pd.read_csv(out, keep_default_na=False)
    assert list(exported.columns) == ["question", "times_asked", "bot_matched", "expected_question"]
    assert "What is y?" in set(exported["bot_matched"])


def test_feedback_cli(tmp_path, capsys):
    store = fb.FeedbackStore(tmp_path / "f.jsonl"); store.add(fb.build_entry("how to bake", "declined", None, 0.0, False))
    assert fb.main(["--path", str(store.path), "--export", str(tmp_path / "e.csv")]) == 0
    assert "how to bake" in capsys.readouterr().out and (tmp_path / "e.csv").exists()
    assert fb.main(["--path", str(tmp_path / "missing.jsonl")]) == 0


# -------------------------------------------------------------------- review
@pytest.fixture()
def faq_df():
    return pd.DataFrame([["What is a?", "Answer A.", "T1", "definition", ""], ["Define a", "Answer A.", "T1", "definition", ""],
                         ["What is b?", "Answer B.", "T2", "definition", "yes"], ["What is c?", "Answer C.", "T2", "definition", ""]],
                        columns=["question", "answer", "topic", "answer_type", "reviewed"])


def test_review_stats_and_export(faq_df):
    s = rv.stats(faq_df)
    assert int(s.loc["T2", "reviewed"]) == 1 and int(s.loc["T1", "percent"]) == 0
    todo = rv.export_todo(faq_df)
    assert list(todo["question"]) == ["What is a?", "What is c?"]        # phrasings share an answer; reviewed row excluded
    assert len(rv.export_todo(faq_df, include_reviewed=True)) == 3 and len(rv.export_todo(faq_df, subject="T2")) == 1


def test_review_apply_ok_fix_and_problems(faq_df):
    review = pd.DataFrame([["What is a?", "ok", ""], ["What is c?", "fix", "Better C."], ["What is zzz?", "ok", ""],
                           ["What is b?", "fix", ""], ["What is b?", "", ""]], columns=["question", "decision", "corrected_answer"])
    new, rep = rv.apply_review(faq_df, review)
    assert rep["marked_ok"] == 2 and rep["fixed"] == 1 and rep["unknown_questions"] == ["What is zzz?"]
    assert rep["fix_without_text"] == ["What is b?"] and rep["skipped_blank"] == 1
    assert list(new["reviewed"]) == ["yes", "yes", "yes", "yes"] and new.loc[3, "answer"] == "Better C."
    assert set(new.loc[[0, 1], "answer"]) == {"Answer A."}
    with pytest.raises(DatasetError):
        rv.apply_review(faq_df, pd.DataFrame({"foo": [1]}))


def test_review_cli_roundtrip(tmp_path, faq_df, capsys):
    faq_path = tmp_path / "faq.csv"; faq_df.to_csv(faq_path, index=False)
    todo = tmp_path / "todo.csv"
    assert rv.main(["--faq", str(faq_path), "export", "--out", str(todo)]) == 0
    t = pd.read_csv(todo, keep_default_na=False); t["decision"] = "ok"; t.to_csv(todo, index=False)
    assert rv.main(["--faq", str(faq_path), "apply", str(todo)]) == 0
    assert (tmp_path / "faq.csv.bak").exists() and (pd.read_csv(faq_path, keep_default_na=False)["reviewed"] == "yes").all()
    assert rv.main(["--faq", str(faq_path), "stats"]) == 0 and "4 of 4" in capsys.readouterr().out


def test_cleaning_keeps_and_normalises_reviewed_column(tmp_path):
    from ml.preprocess import load_clean_faq
    p = tmp_path / "f.csv"
    pd.DataFrame({"question": ["q one?", "q two?"], "answer": ["a1", "a2"], "topic": ["T", "T"], "reviewed": ["Yes", "maybe"]}).to_csv(p, index=False)
    df, _ = load_clean_faq(p)
    assert list(df["reviewed"]) == ["yes", ""]


def test_about_page_reports_review_progress(tiny_faq_path, tmp_path):
    df = pd.read_csv(tiny_faq_path); df["reviewed"] = ["yes", "", "", "yes"]; df.to_csv(tiny_faq_path, index=False)
    c = create_app({"TESTING": True, "FAQ_PATH": str(tiny_faq_path), "RETRIEVER_ARTIFACT": str(tmp_path / "n.joblib")}).test_client()
    assert b"2 of 4" in c.get("/about").data


# ---------------------------------------------------------------- add_phrasings
def test_add_phrasings_adds_refuses_and_keeps_answers(faq_df):
    aliases = pd.DataFrame([["tell me about a", "What is a?"], ["WHAT IS B", "What is b?"], ["what is b", "What is b?"],
                            ["brand new wording", "No such question?"], ["held out question", "What is c?"], ["", "What is a?"]],
                           columns=["alias", "question"])
    new, rep = ap.add_phrasings(faq_df, aliases, protected={"held out question"})
    reasons = dict(rep["refused"])
    assert rep["added"] == 1 and len(new) == len(faq_df) + 1
    assert new.iloc[-1]["question"] == "tell me about a" and new.iloc[-1]["answer"] == "Answer A." and new.iloc[-1]["topic"] == "T1"
    assert "already" in reasons["WHAT IS B"] and "not found" in reasons["brand new wording"]
    assert "contaminate" in reasons["held out question"] and "empty" in reasons[""]


def test_add_phrasings_cli_and_protected_keys(tmp_path, monkeypatch, faq_df, capsys):
    (tmp_path / "test_questions_x.csv").write_text("question\nsecret test q\n", encoding="utf-8")
    assert ap.protected_keys(tmp_path) == {"secret test q"}
    faq_path = tmp_path / "faq.csv"; faq_df.to_csv(faq_path, index=False)
    inp = tmp_path / "new.csv"; pd.DataFrame({"alias": ["explain a to me"], "question": ["What is a?"]}).to_csv(inp, index=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert ap.main(["--input", str(inp), "--faq", str(faq_path)]) == 0
    assert "Added 1" in capsys.readouterr().out and (tmp_path / "faq.csv.bak").exists() and len(pd.read_csv(faq_path)) == 5
    assert ap.main(["--input", str(tmp_path / "missing.csv"), "--faq", str(faq_path)]) == 1


# ------------------------------------------------------------------- coverage
def test_coverage_counts_concepts_and_phrasings(faq_df):
    c = cov.coverage(faq_df).set_index("subject")
    assert c.loc["T1", "questions"] == 2 and c.loc["T1", "concepts"] == 1 and c.loc["T1", "single_phrasing"] == 0
    assert c.loc["T2", "single_phrasing"] == 2
    report = cov.build_report(cov.coverage(faq_df))
    assert "(thin)" in report and "T1" in report


# -------------------------------------------------------------- make_test_file
def test_make_test_file(faq_df):
    labelled = pd.DataFrame([["How is a defined?", "what is a?"], ["Who painted the Mona Lisa?", "none"], ["Which one is b exactly", "What is b?|What is c?"],
                             ["What is b?", "What is b?"], ["How is a defined?", "What is a?"]], columns=["question", "expected_question"])
    df, rep = mt.build_test_df(labelled, faq_df)
    assert list(df["question"]) == ["How is a defined?", "Who painted the Mona Lisa?", "Which one is b exactly"]
    assert list(df["in_scope"]) == ["yes", "no", "yes"] and df.loc[0, "expected_question"] == "What is a?" and df.loc[0, "expected_topic"] == "T1"
    assert df.loc[2, "expected_question"] == "What is b?|What is c?" and set(df["split"]) == {"test"}
    assert rep["dropped_leaky"] == ["What is b?"] and len(rep["dropped_duplicates"]) == 1
    with pytest.raises(DatasetError):
        mt.build_test_df(pd.DataFrame({"question": ["x y"], "expected_question": ["No such stored question?"]}), faq_df)
    with pytest.raises(DatasetError):
        mt.build_test_df(pd.DataFrame({"question": ["x"]}), faq_df)


def test_generated_test_file_works_with_evaluate(tmp_path, faq_df):
    from ml.evaluate import load_test_set
    df, _ = mt.build_test_df(pd.DataFrame({"question": ["How is a defined?"], "expected_question": ["What is a?"]}), faq_df)
    path = tmp_path / "real.csv"; df.to_csv(path, index=False)
    assert len(load_test_set(path)) == 1


# ------------------------------------------------------------ regression check
def test_regression_compare_flags_real_drops_only():
    base = {"A": {"top1_exact_question_rate": 0.8, "top3_exact_question_rate": 0.94, "answer_precision": 0.86,
                  "helped_within_one_click_rate": 0.83, "false_match_count": 3, "wrong_answer_count": 10}}
    same = json.loads(json.dumps(base))
    assert rc.compare(same, base) == []
    noise = json.loads(json.dumps(base)); noise["A"]["top3_exact_question_rate"] = 0.92; noise["A"]["false_match_count"] = 5
    assert rc.compare(noise, base) == []
    worse = json.loads(json.dumps(base)); worse["A"]["top3_exact_question_rate"] = 0.85; worse["A"]["false_match_count"] = 9
    problems = rc.compare(worse, base)
    assert len(problems) == 2 and any("top3" in p for p in problems) and any("false_match" in p for p in problems)
    assert rc.compare({}, base) == ["set A: missing from current run"]


def test_regression_check_passes_on_committed_baseline():
    assert rc.main([]) == 0


def test_regression_baseline_update(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "BASELINE_PATH", tmp_path / "base.json")
    assert rc.main(["--update-baseline"]) == 0 and (tmp_path / "base.json").exists()
    assert rc.main([]) == 0


# ----------------------------------------------------------------------- quiz
def test_quiz_page_and_flashcard_api(client):
    assert b"Flashcard quiz" in client.get("/quiz").data
    first = client.get("/api/flashcard?seed=1").get_json()
    assert first["done"] is False and {"id", "question", "answer", "topic", "remaining"} <= set(first)
    assert client.get("/api/flashcard?seed=1").get_json() == first                  # seed makes it repeatable
    seen = []
    for _ in range(4):
        card = client.get("/api/flashcard?exclude=" + ",".join(map(str, seen))).get_json()
        assert card["id"] not in seen
        seen.append(card["id"])
    assert client.get("/api/flashcard?exclude=" + ",".join(map(str, seen))).get_json() == {"done": True, "remaining": 0}


@pytest.mark.parametrize("query", ["subject=Nope", "exclude=1;2", "exclude=abc", "exclude=" + "1," * 3000])
def test_flashcard_rejects_bad_input(client, query):
    assert client.get("/api/flashcard?" + query).status_code == 400


def test_flashcard_subject_filter(client):
    assert client.get("/api/flashcard?subject=Fruit&seed=2").get_json()["topic"] == "Fruit"


def test_flashcards_skip_duplicate_answers():
    from ml.predict import Retriever
    df = pd.DataFrame([["What is a?", "Same.", "T", "d"], ["Define a", "Same.", "T", "d"], ["What is b?", "Other.", "T", "d"]],
                      columns=["question", "answer", "topic", "answer_type"])
    assert len(Retriever().fit(df).flashcards()) == 2


# ------------------------------------------------------ Taglish / wording tweaks
def test_taglish_and_instruction_words_are_filler():
    assert normalize_text("Ano ang deadlock?") == normalize_text("deadlock")
    assert normalize_text("Paano po ang paging, pwede ba?") == normalize_text("paging")
    assert normalize_text("define deadlock") == normalize_text("explain deadlock") == normalize_text("describe a deadlock") == "deadlock"


def test_versus_means_difference_between():
    assert normalize_text("stack vs queue") == normalize_text("stack versus queue")
    assert "difference" in normalize_text("tcp vs udp")


def test_real_questions_with_new_wording_rules():
    from ml.predict import Retriever
    from ml.preprocess import load_clean_faq
    r = Retriever().fit(load_clean_faq()[0])
    assert r.answer("ano ang deadlock")["matched_question"] == "What is a deadlock?"
    assert r.answer("define deadlock")["matched_question"] == "What is a deadlock?"
    assert r.answer("tcp vs udp")["matched_question"] == "What is the difference between TCP and UDP?"


def test_evaluation_handles_test_file_without_in_scope_questions(tmp_path, monkeypatch, tiny_faq_df):
    """Found by the pilot dry run: a file of only refusals used to crash the classifier step."""
    import ml.evaluate as ev
    df = pd.DataFrame([["who painted the mona lisa", "", "", "no", "test", "easy"], ["how do planes fly", "", "", "no", "test", "easy"]],
                      columns=["question", "expected_topic", "expected_question", "in_scope", "split", "difficulty"])
    intents = pd.DataFrame({"text": ["a b", "c d"], "intent": ["Fruit", "Space"]})
    res = ev.run_evaluation(tiny_faq_df, intents, df, threshold=0.2)
    assert res["classifier_logreg"] is None and res["retrieval_test_at_configured_threshold"]["questions_in_scope"] == 0
    assert "Skipped" in ev.build_report(res)
    path = tmp_path / "refusals.csv"; df.to_csv(path, index=False)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    assert ev.main(["--test-file", str(path), "--tag", "x"]) == 0


def test_evaluation_handles_test_file_without_out_of_scope_questions(tiny_faq_df):
    import ml.evaluate as ev
    df = pd.DataFrame([["rocket space", "Space", "What is a rocket?", "yes", "test", "easy"]],
                      columns=["question", "expected_topic", "expected_question", "in_scope", "split", "difficulty"])
    intents = pd.DataFrame({"text": ["fruit a", "space b"], "intent": ["Fruit", "Space"]})
    res = ev.run_evaluation(tiny_faq_df, intents, df, threshold=0.2)
    assert res["retrieval_test_at_configured_threshold"]["no_match_correct_rejection_rate"] is None
    assert "n/a" in ev.build_report(res)


# ---------------------------------------------------------------- pilot kit
def test_pilot_status_counts(tiny_faq_df):
    from ml import pilot_status as ps
    topic_of = dict(zip(tiny_faq_df["question"], tiny_faq_df["topic"]))
    entries = [fb.build_entry("What is a banana", "up", "What is a banana?", 0.9, True), fb.build_entry("what is a banana?", "up", "What is a banana?", 0.9, True),
               fb.build_entry("what is a rocket", "down", "What is a rocket?", 0.7, True), fb.build_entry("how do planes fly", "declined", None, 0.0, False)]
    s = ps.status(entries, topic_of, goal=3)
    assert (s["entries"], s["distinct_questions"], s["subjects_touched"]) == (4, 3, 2)
    assert (s["helpful"], s["not_helpful"], s["unanswered_distinct"]) == (2, 1, 1)
    assert s["goal_reached"] is True and s["enough_variety"] is False


def test_pilot_status_cli(tmp_path, capsys):
    from ml import pilot_status as ps
    store = fb.FeedbackStore(tmp_path / "f.jsonl"); store.add(fb.build_entry("what is a deadlock", "up", "What is a deadlock?", 0.7, True))
    assert ps.main(["--path", str(store.path), "--goal", "5"]) == 0
    out = capsys.readouterr().out
    assert "distinct questions: 1 of goal 5" in out and "Keep collecting" in out
    assert ps.main(["--path", str(tmp_path / "none.jsonl")]) == 0


def test_host_defaults_to_this_computer_and_network_hosting_is_guarded(monkeypatch):
    import importlib
    from app import check_host
    monkeypatch.delenv("STUDYBUDDY_HOST", raising=False)
    importlib.reload(config)
    assert config.HOST == "127.0.0.1"
    assert check_host("127.0.0.1", False) == "" and check_host("localhost", True) == ""
    assert "other computers" in check_host("0.0.0.0", False)
    with pytest.raises(SystemExit, match="debug"):
        check_host("0.0.0.0", True)


def test_pilot_kit_files_are_usable():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "pilot"
    for name in ("PILOT_GUIDE.md", "student_notice.txt", "question_collection_form.md", "labelling_template.csv"):
        assert (root / name).exists() and (root / name).stat().st_size > 0, name
    assert (root / "labelling_template.csv").read_text().strip() == "question,expected_question,difficulty"
    notice = (root / "student_notice.txt").read_text()
    assert "do NOT save your name" in notice and "IP address" in notice
    # every command the guide tells people to run must be a real module
    import importlib, re
    for module in set(re.findall(r"python -m (ml\.[a-z_]+)", (root / "PILOT_GUIDE.md").read_text())):
        importlib.import_module(module)


# ------------------------------------------------- cloud-sync privacy guard (found on a real Windows setup)
@pytest.mark.parametrize("path", [r"C:\Users\ecen2\OneDrive\Documents\student_study_assistant\data\feedback\feedback.jsonl",
                                  "/home/u/Dropbox/proj/feedback.jsonl", "/Users/u/Google Drive/x/f.jsonl", "/Users/u/iCloud Drive/f.jsonl"])
def test_cloud_synced_paths_trigger_a_privacy_warning(path):
    from app import sync_folder_warning
    assert "PRIVACY WARNING" in sync_folder_warning(path) and "STUDYBUDDY_FEEDBACK_DIR" in sync_folder_warning(path)


@pytest.mark.parametrize("path", ["C:\\StudyBuddyData\\feedback.jsonl", "/tmp/feedback.jsonl", "/home/u/projects/feedback.jsonl"])
def test_local_paths_do_not_warn(path):
    from app import sync_folder_warning
    assert sync_folder_warning(path) == ""


def test_app_logs_the_warning_and_about_page_is_honest(tiny_faq_path, tmp_path, caplog):
    synced = tmp_path / "OneDrive" / "fb"
    with caplog.at_level("WARNING"):
        c = create_app({"TESTING": True, "FAQ_PATH": str(tiny_faq_path), "RETRIEVER_ARTIFACT": str(tmp_path / "n.joblib"),
                        "FEEDBACK_PATH": str(synced / "f.jsonl")}).test_client()
    assert "PRIVACY WARNING" in caplog.text
    assert b"cloud-synced folder" in c.get("/about").data
    local = create_app({"TESTING": True, "FAQ_PATH": str(tiny_faq_path), "RETRIEVER_ARTIFACT": str(tmp_path / "n.joblib"),
                        "FEEDBACK_PATH": str(tmp_path / "plain" / "f.jsonl")}).test_client()
    assert b"It stays on this computer" in local.get("/about").data


def test_feedback_dir_can_be_moved_with_an_environment_variable(monkeypatch, tmp_path):
    import importlib
    monkeypatch.setenv("STUDYBUDDY_FEEDBACK_DIR", str(tmp_path / "elsewhere"))
    importlib.reload(config)
    assert config.FEEDBACK_PATH == tmp_path / "elsewhere" / "feedback.jsonl"
    monkeypatch.delenv("STUDYBUDDY_FEEDBACK_DIR")
    importlib.reload(config)
    assert config.FEEDBACK_PATH.name == "feedback.jsonl" and "elsewhere" not in str(config.FEEDBACK_PATH)


# ----------------------------- guards added after a real session: leftover env settings corrupted a baseline
def test_active_overrides_detects_scoring_variables(monkeypatch):
    import importlib
    for k in ("STUDYBUDDY_THRESHOLD", "STUDYBUDDY_CHAR_WEIGHT", "STUDYBUDDY_ANSWER_WEIGHT"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(config)
    assert config.active_overrides() == {}
    monkeypatch.setenv("STUDYBUDDY_THRESHOLD", "0.20"); monkeypatch.setenv("STUDYBUDDY_PORT", "5050")
    assert config.active_overrides() == {"STUDYBUDDY_THRESHOLD": "0.20"}          # port is not a scoring setting
    monkeypatch.delenv("STUDYBUDDY_THRESHOLD")
    importlib.reload(config)


def test_regression_check_refuses_to_run_or_update_baseline_with_overrides(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rc, "BASELINE_PATH", tmp_path / "base.json")
    monkeypatch.setenv("STUDYBUDDY_THRESHOLD", "0.20")
    assert rc.main(["--update-baseline"]) == 2 and not (tmp_path / "base.json").exists()
    assert rc.main([]) == 2
    out = capsys.readouterr().out
    assert "STUDYBUDDY_THRESHOLD=0.20" in out and "NEW PowerShell window" in out


def test_evaluation_report_flags_overridden_settings(tiny_faq_df, monkeypatch):
    import ml.evaluate as ev
    df = pd.DataFrame([["rocket space", "Space", "What is a rocket?", "yes", "test", "easy"]],
                      columns=["question", "expected_topic", "expected_question", "in_scope", "split", "difficulty"])
    intents = pd.DataFrame({"text": ["fruit a", "space b"], "intent": ["Fruit", "Space"]})
    monkeypatch.setenv("STUDYBUDDY_THRESHOLD", "0.20")
    res = ev.run_evaluation(tiny_faq_df, intents, df, threshold=0.2)
    assert res["scoring_overrides"] == {"STUDYBUDDY_THRESHOLD": "0.20"}
    assert "overridden by environment variables" in ev.build_report(res)
    monkeypatch.delenv("STUDYBUDDY_THRESHOLD")
    assert "overridden" not in ev.build_report(ev.run_evaluation(tiny_faq_df, intents, df, threshold=0.2))


def test_review_apply_with_blank_decisions_does_not_touch_the_knowledge_base(tmp_path, capsys):
    faq_path = tmp_path / "faq.csv"
    pd.DataFrame({"question": ["What is a?"], "answer": ["A."], "topic": ["T"], "answer_type": ["definition"]}).to_csv(faq_path, index=False)
    before = faq_path.read_bytes()
    assert rv.main(["--faq", str(faq_path), "export", "--out", str(tmp_path / "todo.csv")]) == 0
    assert rv.main(["--faq", str(faq_path), "apply", str(tmp_path / "todo.csv")]) == 0
    assert faq_path.read_bytes() == before and not (tmp_path / "faq.csv.bak").exists()
    assert "Nothing to apply" in capsys.readouterr().out


def test_tests_run_with_default_settings_even_if_the_shell_has_overrides():
    assert not [k for k in __import__("os").environ if k.startswith("STUDYBUDDY_")]
    assert config.SIMILARITY_THRESHOLD == 0.25 and config.ANSWER_WEIGHT == 0.65
