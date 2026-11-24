"""Build a 15x15 supervised dataset (board -> next move) from Gomocup 2019 Freestyle games.

Gomocup Freestyle is played on 20x20. Games whose stones all fit inside a 15x15 window
(~89% of them) are shifted onto a 15x15 board; the rest are dropped.

Each sample is the position *before* a move, seen from the side to move:
    X[i, 0] = stones of the player to move, X[i, 1] = opponent stones   (uint8, 0/1)
    y[i]    = index of the move played = row * 15 + col                  (0..224)
    z[i]    = game result for the player to move: +1 win, -1 loss, 0 unknown (no five on the board)
Opening stones (the fixed tournament openings, logged with time 0) are placed on the board
but never used as labels, since no engine chose them.

Usage:
    python create_dataset.py                       # downloads the zip if missing
    python create_dataset.py --winner-only         # only learn from the winner's moves
"""
import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://gomocup.org/static/tournaments/2019/results/gomocup2019results.zip"
N = 15
MOVE_RE = re.compile(r"^\s*(\d+),(\d+),(-?\d+)\s*$")


def parse_psq(text):
    """Return list of (x, y, time_ms) moves, 0-based coords."""
    moves = []
    for line in text.splitlines()[1:]:  # line 0 is the "Piskvorky 20x20, ..." header
        m = MOVE_RE.match(line)
        if not m:
            break
        moves.append((int(m[1]) - 1, int(m[2]) - 1, int(m[3])))
    return moves


def has_five(stones, x, y):
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for s in (1, -1):
            k = 1
            while (x + s * k * dx, y + s * k * dy) in stones:
                count += 1
                k += 1
        if count >= 5:
            return True
    return False


def game_to_samples(moves, winner_only):
    xs = [m[0] for m in moves]
    ys = [m[1] for m in moves]
    w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    if w > N or h > N:
        return None
    # center the bounding box on the 15x15 board
    ox = min(xs) - (N - w) // 2
    oy = min(ys) - (N - h) // 2
    moves = [(x - ox, y - oy, t) for x, y, t in moves]

    opening = 0
    while opening < len(moves) and moves[opening][2] == 0:
        opening += 1

    winner = None  # 0 = first player (black), 1 = second
    last = len(moves) - 1
    lx, ly, _ = moves[last]
    if has_five({(x, y) for i, (x, y, _) in enumerate(moves) if i % 2 == last % 2}, lx, ly):
        winner = last % 2
    if winner_only and winner is None:
        return None

    board = np.zeros((2, N, N), np.uint8)  # [black, white], indexed [row=y, col=x]
    X, Y, Z = [], [], []
    for i, (x, y, _) in enumerate(moves):
        me = i % 2
        if board[:, y, x].any():  # corrupt record: occupied square
            return None
        if i >= opening and (not winner_only or me == winner):
            X.append(board[[me, 1 - me]].copy())
            Y.append(y * N + x)
            Z.append(0 if winner is None else (1 if me == winner else -1))
        board[me, y, x] = 1
    return X, Y, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/gomocup2019results.zip")
    ap.add_argument("--out", default="data/gomoku_15x15.npz")
    ap.add_argument("--leagues", default="Freestyle1,Freestyle2,Freestyle3,Freestyle4")
    ap.add_argument("--winner-only", action="store_true")
    args = ap.parse_args()

    zpath = Path(args.zip)
    if not zpath.exists():
        zpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)

    leagues = tuple(l + "/" for l in args.leagues.split(","))
    X, Y, Z, G = [], [], [], []
    kept = dropped = 0
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if n.startswith(leagues) and n.endswith(".psq"))
        for name in names:
            moves = parse_psq(io.TextIOWrapper(z.open(name), encoding="latin1").read())
            res = game_to_samples(moves, args.winner_only) if len(moves) >= 5 else None
            if res is None or not res[0]:
                dropped += 1
                continue
            X += res[0]
            Y += res[1]
            Z += res[2]
            G += [kept] * len(res[1])  # game id, for a leak-free train/val split
            kept += 1

    X = np.stack(X)
    Y = np.array(Y, np.int16)
    Z = np.array(Z, np.int8)
    G = np.array(G, np.int32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=Y, z=Z, game=G)
    print(f"games kept {kept}, dropped {dropped}; samples {len(Y)}; X {X.shape} -> {out}")


if __name__ == "__main__":
    main()
