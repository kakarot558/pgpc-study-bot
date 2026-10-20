"""Shortcut:  python -m ml.feedback_report [--export data/feedback/needs_label.csv]"""
import sys

from ml.feedback import main

if __name__ == "__main__":
    sys.exit(main())
