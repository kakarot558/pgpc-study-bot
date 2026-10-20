"""Local feedback log: which answers helped and which questions the bot could not answer.

PRIVACY: an entry is written only when a student clicks "Helpful" / "Not helpful", or - if the
owner switches it on with STUDYBUDDY_LOG_DECLINED=1 - when the bot declines a question. Entries hold
the question text, the matched question, the score, a rating and a timestamp. No names, IP
addresses or browser details are stored, and the file never leaves this computer.

Review it with:   python -m ml.feedback_report
"""
import argparse
import csv
import json
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import config
from ml.preprocess import normalize_whitespace, question_key

RATINGS = ("up", "down", "declined")


class FeedbackError(ValueError):
    """Raised for invalid feedback or when the log is full."""


def build_entry(message, rating, matched_question=None, similarity=None, match_found=None,
                max_len: int = config.MAX_MESSAGE_LENGTH) -> dict:
    """Validate the pieces and return a clean dict ready to store."""
    if rating not in RATINGS:
        raise FeedbackError("rating must be 'up' or 'down'.")
    if not isinstance(message, str) or not normalize_whitespace(message):
        raise FeedbackError("message must be non-empty text.")
    message = normalize_whitespace(message)
    if len(message) > max_len:
        raise FeedbackError(f"message must be at most {max_len} characters.")
    if matched_question is not None:
        if not isinstance(matched_question, str) or len(matched_question) > 300:
            raise FeedbackError("matched_question must be text of at most 300 characters.")
        matched_question = normalize_whitespace(matched_question) or None
    if similarity is not None:
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)) or not 0 <= similarity <= 2:
            raise FeedbackError("similarity must be a number between 0 and 2.")
        similarity = round(float(similarity), 4)
    if match_found is not None and not isinstance(match_found, bool):
        raise FeedbackError("match_found must be true or false.")
    return {"time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "rating": rating, "message": message,
            "matched_question": matched_question, "similarity": similarity, "match_found": match_found}


class FeedbackStore:
    """Append-only JSON-lines file, safe for several threads."""

    def __init__(self, path=config.FEEDBACK_PATH, max_bytes: int = config.FEEDBACK_MAX_BYTES):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def add(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            if self.path.exists() and self.path.stat().st_size + len(line.encode("utf-8")) > self.max_bytes:
                raise FeedbackError("The feedback log is full. Archive or delete the file first.")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def read_all(self) -> list:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue                                # ignore a damaged line
            if isinstance(item, dict) and item.get("rating") in RATINGS and isinstance(item.get("message"), str):
                entries.append(item)
        return entries


def summarize(entries: list) -> dict:
    """Counts plus the lists an owner needs: what to add, what to fix."""
    ratings = Counter(e["rating"] for e in entries)
    down_by_match, declined, down_unmatched, down_detail = Counter(), {}, {}, {}
    for e in entries:
        key = question_key(e["message"])
        if e["rating"] == "down":
            if e.get("matched_question"):
                down_by_match[e["matched_question"]] += 1
                down_detail.setdefault(key, [e["message"], e["matched_question"], 0])[2] += 1
            else:
                down_unmatched.setdefault(key, [e["message"], 0])[1] += 1
        elif e["rating"] == "declined" or (e["rating"] == "up" and e.get("match_found") is False):
            declined.setdefault(key, [e["message"], 0])[1] += 1
    return {"total": len(entries), "up": ratings["up"], "down": ratings["down"], "declined": ratings["declined"],
            "answers_marked_unhelpful": down_by_match.most_common(),
            "unanswered_questions": sorted(((m, c) for m, c in declined.values()), key=lambda x: -x[1]),
            "unhelpful_without_match": sorted(((m, c) for m, c in down_unmatched.values()), key=lambda x: -x[1]),
            "unhelpful_answers_detail": sorted(((m, q, c) for m, q, c in down_detail.values()), key=lambda x: -x[2])}


def export_needs_label(summary: dict, out_path) -> int:
    """Write questions an instructor should label (feeds ml.make_test_file / add_phrasings):
    unanswered ones, and ones whose answer was rated 'Not helpful' (bot_matched shows what it wrongly returned)."""
    rows = [(m, c, "") for m, c in summary["unanswered_questions"] + summary["unhelpful_without_match"]]
    rows += [(m, c, q) for m, q, c in summary["unhelpful_answers_detail"]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "times_asked", "bot_matched", "expected_question"])
        for msg, count, matched in rows:
            w.writerow([msg, count, matched, ""])
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise the local feedback log.")
    ap.add_argument("--path", default=str(config.FEEDBACK_PATH))
    ap.add_argument("--export", default=None, help="write unanswered questions to this CSV for labelling")
    args = ap.parse_args(argv)
    entries = FeedbackStore(args.path).read_all()
    if not entries:
        print("No feedback recorded yet.")
        return 0
    s = summarize(entries)
    print(f"{s['total']} entries: {s['up']} helpful, {s['down']} not helpful, {s['declined']} declined questions logged")
    if s["answers_marked_unhelpful"]:
        print("\nAnswers marked 'not helpful' (review or fix these first):")
        for q, c in s["answers_marked_unhelpful"][:10]:
            print(f"  {c:>3} x {q}")
    if s["unanswered_questions"] or s["unhelpful_without_match"]:
        print("\nQuestions the bot could not answer (candidates to add):")
        for q, c in (s["unanswered_questions"] + s["unhelpful_without_match"])[:15]:
            print(f"  {c:>3} x {q}")
    if args.export:
        print(f"\nWrote {export_needs_label(s, args.export)} rows to {args.export}. Fill the 'expected_question' column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
