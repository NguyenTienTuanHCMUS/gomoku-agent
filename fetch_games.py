"""Download games that players sent from the web page (GitHub Issues titled "[game] ...").

Needs the GitHub CLI (gh) logged in. Writes one JSON object per game, e.g.
{"issue": 12, "result": "human", "human": "X", "sims": 100, "undos": 0, "model": "pv1", "moves": ["h8", "i9", ...]}

    python fetch_games.py --repo NguyenTienTuanHCMUS/gomoku-agent
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

BLOCK = re.compile(r"```gomoku\s*\n(.*?)```", re.S)
MOVE = re.compile(r"^[a-o](1[0-5]|[1-9])$")


def parse(body):
    m = BLOCK.search(body or "")
    if not m:
        return None
    fields = dict(line.split(":", 1) for line in m[1].strip().splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    moves = fields.get("moves", "").split()
    if not moves or not all(MOVE.match(x) for x in moves) or len(set(moves)) != len(moves):
        return None
    return {
        "result": fields.get("result"), "human": fields.get("human"), "sims": int(fields.get("sims", 0) or 0),
        "undos": int(fields.get("undos", 0) or 0), "model": fields.get("model"), "moves": moves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="NguyenTienTuanHCMUS/gomoku-agent")
    ap.add_argument("--out", default="data/human_games.jsonl")
    args = ap.parse_args()

    raw = subprocess.run(
        ["gh", "issue", "list", "--repo", args.repo, "--state", "all", "--limit", "5000",
         "--search", '"[game]" in:title', "--json", "number,title,body"],
        check=True, capture_output=True, text=True, encoding="utf-8").stdout
    games, skipped = [], 0
    for issue in json.loads(raw):
        g = parse(issue["body"])
        if g is None:
            skipped += 1
            continue
        games.append({"issue": issue["number"], **g})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for g in sorted(games, key=lambda g: g["issue"]):
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    wins = sum(g["result"] == "human" for g in games)
    print(f"{len(games)} games ({wins} won by the human), {skipped} issues skipped -> {args.out}")


if __name__ == "__main__":
    main()
