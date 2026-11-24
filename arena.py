"""Play two agents against each other from real Gomocup openings and report the score.

Each opening is played twice with colours swapped, so neither side profits from a lucky opening.
An agent is "MODEL_PATH:SIMS"; SIMS = 0 means no search (network + win/block check).

    python arena.py --a models/pv1/gomoku_pv.pt:200 --b models/run1/gomoku_policy.pt:0 --openings 30
"""
import argparse
import math
import os
import time
from multiprocessing import Pool

import numpy as np

AGENTS = {}


def make_agent(spec):
    from gomoku import MCTS, load_model, policy_move
    path, sims = spec.rsplit(":", 1)
    model = load_model(path, threads=1)
    if int(sims) == 0:
        return lambda b, p: policy_move(model, b, p)[0]
    search = MCTS(model, sims=int(sims))
    return lambda b, p: search.best_move(b, p)[0]


def openings(data, count, seed):
    """Opening positions = the first labelled position of each game (stones the tournament placed)."""
    d = np.load(data)
    X, game = d["X"], d["game"]
    first = np.unique(game, return_index=True)[1]
    seen, out = set(), []
    for i in first:
        own, opp = X[i, 0].astype(np.int8), X[i, 1].astype(np.int8)
        to_move = 1 if own.sum() == opp.sum() else 2
        b = own * to_move + opp * (3 - to_move)
        key = b.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(b)
    rng = np.random.default_rng(seed)
    return [out[i] for i in rng.choice(len(out), size=min(count, len(out)), replace=False)]


def play(task):
    from gomoku import is_five
    a_spec, b_spec, board, a_is_x = task
    for spec in (a_spec, b_spec):
        if spec not in AGENTS:
            AGENTS[spec] = make_agent(spec)
    board = board.copy()
    player = 1 if (board == 1).sum() == (board == 2).sum() else 2
    a_player = 1 if a_is_x else 2
    spec = {a_player: a_spec, 3 - a_player: b_spec}
    while (board == 0).any():
        r, c = AGENTS[spec[player]](board, player)
        assert board[r, c] == 0
        board[r, c] = player
        if is_five(board, r, c):
            return (1.0 if player == a_player else 0.0), a_is_x, int((board > 0).sum())
        player = 3 - player
    return 0.5, a_is_x, 225


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="MODEL:SIMS")
    ap.add_argument("--b", required=True, help="MODEL:SIMS")
    ap.add_argument("--openings", type=int, default=30)
    ap.add_argument("--data", default="data/gomoku_15x15.npz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    ops = openings(args.data, args.openings, args.seed)
    tasks = [(args.a, args.b, b, a_is_x) for b in ops for a_is_x in (True, False)]
    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for i, (score, a_is_x, stones) in enumerate(pool.imap(play, tasks), 1):
            results.append((score, a_is_x))
            print(f"game {i:3d}/{len(tasks)}  A plays {'X' if a_is_x else 'O'}  "
                  f"{'A wins' if score == 1 else 'B wins' if score == 0 else 'draw'}  ({stones} stones)", flush=True)

    s = np.array([r[0] for r in results])
    ax = np.array([r[1] for r in results])
    score = s.mean()
    se = s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else 0
    clip = min(max(score, 0.01), 0.99)
    elo = 400 * math.log10(clip / (1 - clip))
    print(f"\nA = {args.a}\nB = {args.b}")
    print(f"A score {score:.3f} ± {1.96 * se:.3f} (95%) over {len(s)} games  "
          f"[as X {s[ax].mean():.3f}, as O {s[~ax].mean():.3f}]  ≈ {elo:+.0f} Elo  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
