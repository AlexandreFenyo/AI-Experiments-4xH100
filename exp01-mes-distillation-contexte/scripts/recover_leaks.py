"""Recover the teacher answers set aside by the leak filter when their only "leak" is the phrase
« base de connaissances » (the teacher admitting that the FAQ does not cover a point). The phrase is replaced by
« documentation officielle de Mon espace santé » (same gender and number, so agreements stay correct), which is a
natural thing for the assistant to say. These answers teach the student to admit what it does not know.

    python recover_leaks.py --tag gemma31_s2   ->  data/train/distill_<tag>_recovered.jsonl
"""

import argparse
import random
import re
from collections import Counter

from common import DATA, SYSTEM_STUDENT, load_jsonl, write_jsonl
from teacher_answers import LEAK, NO_SYS_SHARE

SUBS = [
    (re.compile(r"\b([Mm])a base de connaissances(?: officielle)?"), lambda m: ("L" if m.group(1) == "M" else "l") + "a documentation officielle de Mon espace santé"),
    (re.compile(r"\b([Cc])ette base de connaissances(?: officielle)?"), r"\1ette documentation officielle"),
    (re.compile(r"\bbase de connaissances(?: officielle)?(?: (?:de|du service) (?:« ?)?Mon espace santé(?: ?»)?)?"), "documentation officielle de Mon espace santé"),
]


def clean(text):
    for rx, rep in SUBS:
        text = rx.sub(rep, text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    rnd = random.Random(1)
    train, stats = [], Counter()
    for r in load_jsonl(DATA / "gen" / f"teacher_{args.tag}_leaks.jsonl"):
        answers = [clean(a) for a in r["answers"]]
        if any(LEAK.search(a) for a in answers):          # still a real leak after substitution: drop
            stats["dropped"] += 1
            continue
        in_domain = r["type"] not in ("near", "ood")
        hist = [] if (in_domain and rnd.random() < NO_SYS_SHARE) else [{"role": "system", "content": SYSTEM_STUDENT}]
        for q, a in zip(r["turns"], answers):
            hist = hist + [{"role": "user", "content": q}]
            train.append({"id": r["id"], "type": r["type"], "messages": hist + [{"role": "assistant", "content": a}]})
            hist = hist + [{"role": "assistant", "content": a}]
        stats[r["type"]] += 1
    write_jsonl(DATA / "train" / f"distill_{args.tag}_recovered.jsonl", train)
    print(len(train), "examples recovered:", dict(stats))


if __name__ == "__main__":
    main()
