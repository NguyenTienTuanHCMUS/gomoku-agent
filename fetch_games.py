"""Download games played by people and write them to one JSON-lines file for training.

Sources:
  supabase  games saved automatically by the web page (see supabase/README.md). Needs two environment
            variables on your machine only: SUPABASE_URL and SUPABASE_SECRET_KEY (never commit the secret key).
  github    games sent with the "Send this game" button (GitHub Issues titled "[game] ..."); needs `gh` logged in.

Every game is replayed and dropped if it has a repeated square or a result that does not match the board.
One line per game, e.g.
{"source": "supabase", "id": 12, "result": "human", "human": "X", "sims": 100, "undos": 0, "model": "pv1", "moves": ["h8", "i9", ...]}

    python fetch_games.py                      # both sources that are available
    python fetch_games.py --source supabase
"""
import argparse
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from gomoku import N, is_five

BLOCK = re.compile(r"```gomoku\s*\n(.*?)```", re.S)
MOVE = re.compile(r"^[a-o](1[0-5]|[1-9])$")
COLS = "abcdefghijklmno"


def valid(game):
    """Replay the moves; the recorded result must match what happened on the board."""
    moves = game.get("moves") or []
    if not moves or not all(isinstance(m, str) and MOVE.match(m) for m in moves) or len(set(moves)) != len(moves):
        return False
    if game.get("human") not in ("X", "O") or game.get("result") not in ("human", "model", "draw", "unfinished"):
        return False
    board = np.zeros((N, N), np.int8)
    for k, m in enumerate(moves):
        r, c = int(m[1:]) - 1, COLS.index(m[0])
        board[r, c] = 1 if k % 2 == 0 else 2
        if is_five(board, r, c):
            if k != len(moves) - 1:
                return False                      # play continued after a five
            winner = "X" if k % 2 == 0 else "O"
            return game["result"] == ("human" if winner == game["human"] else "model")
    return game["result"] == "unfinished" or (game["result"] == "draw" and len(moves) == N * N)


def from_github(repo):
    raw = subprocess.run(
        ["gh", "issue", "list", "--repo", repo, "--state", "all", "--limit", "5000",
         "--search", '"[game]" in:title', "--json", "number,body"],
        check=True, capture_output=True, text=True, encoding="utf-8").stdout
    games = []
    for issue in json.loads(raw):
        m = BLOCK.search(issue["body"] or "")
        if not m:
            continue
        f = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in m[1].strip().splitlines() if ":" in line)}
        try:
            games.append({"source": "github", "id": issue["number"], "result": f.get("result"), "human": f.get("human"),
                          "sims": int(f.get("sims") or 0), "undos": int(f.get("undos") or 0),
                          "model": f.get("model"), "moves": f.get("moves", "").split()})
        except ValueError:
            continue
    return games


def from_supabase(url, key):
    games, page = [], 1000
    while True:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/rest/v1/games?select=*&order=id&offset={len(games)}&limit={page}",
            headers={"apikey": key, "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req) as resp:
            rows = json.load(resp)
        games += rows
        if len(rows) < page:
            break
    # A game saved again after "undo" keeps its game_uid. Keep the newest finished row,
    # or the newest row if the game was never finished (rows come in id order).
    latest = {}
    for g in games:
        prev = latest.get(g["game_uid"])
        if prev is None or g["result"] != "unfinished" or prev["result"] == "unfinished":
            latest[g["game_uid"]] = g
    return [{"source": "supabase", "id": g["id"], "date": g["created_at"], "result": g["result"], "human": g["human"],
             "sims": g["sims"], "undos": g["undos"], "model": g["model"], "moves": g["moves"]}
            for g in sorted(latest.values(), key=lambda g: g["id"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["all", "supabase", "github"], default="all")
    ap.add_argument("--repo", default="NguyenTienTuanHCMUS/gomoku-agent")
    ap.add_argument("--out", default="data/human_games.jsonl")
    args = ap.parse_args()

    games = []
    if args.source in ("all", "supabase"):
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SECRET_KEY")
        if url and key:
            games += from_supabase(url, key)
        elif args.source == "supabase":
            raise SystemExit("Set SUPABASE_URL and SUPABASE_SECRET_KEY first (see supabase/README.md).")
        else:
            print("Supabase skipped: SUPABASE_URL / SUPABASE_SECRET_KEY not set.")
    if args.source in ("all", "github"):
        games += from_github(args.repo)

    good = [g for g in games if valid(g)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for g in good:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    wins = sum(g["result"] == "human" for g in good)
    print(f"{len(good)} games kept ({wins} won by a human), {len(games) - len(good)} dropped as invalid -> {args.out}")


if __name__ == "__main__":
    main()
