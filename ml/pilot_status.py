"""Progress of a feedback pilot:   python -m ml.pilot_status [--goal 30]

Counts the distinct questions collected, how many subjects they touched, and how many were
rated helpful, not helpful or unanswered, so you know when you have enough to build a real
test set (see pilot/PILOT_GUIDE.md).
"""
import argparse
import sys

import config
from ml.feedback import FeedbackStore
from ml.preprocess import DatasetError, load_clean_faq, question_key


def status(entries: list, topic_of: dict, goal: int = 30) -> dict:
    distinct = {question_key(e["message"]) for e in entries}
    subjects = {topic_of[e["matched_question"]] for e in entries if e.get("matched_question") in topic_of}
    unanswered = {question_key(e["message"]) for e in entries
                  if e["rating"] == "declined" or e.get("match_found") is False}
    return {"entries": len(entries), "distinct_questions": len(distinct), "subjects_touched": len(subjects),
            "helpful": sum(e["rating"] == "up" for e in entries), "not_helpful": sum(e["rating"] == "down" for e in entries),
            "unanswered_distinct": len(unanswered), "goal": goal, "goal_reached": len(distinct) >= goal,
            "enough_variety": len(subjects) >= 5}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=str(config.FEEDBACK_PATH))
    ap.add_argument("--goal", type=int, default=30)
    args = ap.parse_args(argv)
    try:
        faq, _ = load_clean_faq()
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    s = status(FeedbackStore(args.path).read_all(), dict(zip(faq["question"], faq["topic"])), args.goal)
    print(f"Entries: {s['entries']}   distinct questions: {s['distinct_questions']} of goal {s['goal']}")
    print(f"Subjects touched by answered questions: {s['subjects_touched']} (aim for 5 or more)")
    print(f"Helpful: {s['helpful']}   Not helpful: {s['not_helpful']}   distinct unanswered questions: {s['unanswered_distinct']}")
    if s["goal_reached"] and s["enough_variety"]:
        print("\nEnough to start: run  python -m ml.feedback_report --export data/feedback/needs_label.csv")
    else:
        print("\nKeep collecting" + ("" if s["enough_variety"] else " (ask students about more different subjects)") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
