"""Self-play: the model plays against itself with MCTS and records training targets.

For every position it saves:
  - the board, seen from the player to move (same planes as the Gomocup dataset)
  - pi: how MCTS split its visits over the moves. Search makes better choices than the bare
    network, so training the network towards pi teaches it what the search found.
  - z:  the final result for the player to move (+1 win, -1 loss, 0 draw)

Every game starts from a random Gomocup opening (balanced, and different openings keep games varied),
Dirichlet noise is mixed into the root priors, and the first --temp-moves moves are sampled from pi
instead of always taking the most visited move.

Games run in batches (batch_mcts.py): hundreds of games at once, so the network judges hundreds of
positions per call. Uses the GPU when PyTorch sees one (Kaggle, or a local NVIDIA card), else the CPU.

    python selfplay.py --model models/pv1/gomoku_pv.pt --games 800 --out data/selfplay/gen1.npz
"""
import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np


def worker(job):
    """Plays one share of the games on one device; returns the recorded arrays."""
    import torch
    from batch_mcts import Evaluator, pick_device, play_games
    wid, tasks, a = job
    device = pick_device(wid)
    if device == "cpu":
        torch.set_num_threads(a["threads"])
    ev = Evaluator(a["model"], device)
    t0, count = time.time(), [0]
    stats = {0: 0, 1: 0, 2: 0}
    lengths = []

    def progress(g):
        count[0] += 1
        stats[g.winner] += 1
        lengths.append(len(g.moves))
        if count[0] % a["report"] == 0 or count[0] == len(tasks):
            el = time.time() - t0
            print(f"  [worker {wid} {device}] {count[0]}/{len(tasks)} games  X wins {stats[1]}  O wins {stats[2]}  "
                  f"draws {stats[0]}  avg {np.mean(lengths):.0f} moves  {el / 60:.1f} min, "
                  f"~{el / count[0] * (len(tasks) - count[0]) / 60:.0f} min left", flush=True)

    games = play_games(tasks, [ev], a["sims"], parallel=a["parallel"], noise=(a["alpha"], a["eps"]),
                       temp_moves=a["temp_moves"], reuse=True, record=True, progress=progress)
    X, PI, Z, G = [], [], [], []
    for g in games:
        X.append(np.stack(g.X)); PI.append(np.stack(g.PI)); G.append(np.full(len(g.who), g.id, np.int32))
        Z.append(np.array([0 if g.winner == 0 else (1 if w == g.winner else -1) for w in g.who], np.int8))
    return np.concatenate(X), np.concatenate(PI), np.concatenate(Z), np.concatenate(G), stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pv1/gomoku_pv.pt")
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--out", default="data/selfplay/gen1.npz")
    ap.add_argument("--data", default="data/gomoku_15x15.npz", help="where the Gomocup openings come from")
    ap.add_argument("--temp-moves", type=int, default=6, help="moves sampled from pi at the start of each game")
    ap.add_argument("--alpha", type=float, default=0.15, help="Dirichlet noise concentration")
    ap.add_argument("--eps", type=float, default=0.25, help="share of the root priors replaced by noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="processes (0 = auto: CPU count - 1 on a GPU machine, 1 on CPU only). "
                         "On a GPU the Python tree work is the bottleneck, so more processes help")
    ap.add_argument("--parallel", type=int, default=0, help="games searched at once per process (0 = 256 GPU / 64 CPU)")
    args = ap.parse_args(argv)

    import torch
    gpu = torch.cuda.is_available()
    workers = args.workers or (max(1, (os.cpu_count() or 2) - 1) if gpu else 1)
    parallel = args.parallel or (256 if gpu else 64)

    from arena import all_openings
    ops = all_openings(args.data)
    rng = np.random.default_rng(args.seed)
    picks = rng.integers(len(ops), size=args.games)
    seeds = rng.integers(1 << 62, size=args.games)
    tasks = [(int(s), ops[i], {1: 0, 2: 0}) for s, i in zip(seeds, picks)]
    opts = dict(model=args.model, sims=args.sims, alpha=args.alpha, eps=args.eps, temp_moves=args.temp_moves,
                parallel=parallel, threads=max(1, (os.cpu_count() or 2) // workers),
                report=max(10, args.games // workers // 10))
    jobs = [(w, tasks[w::workers], opts) for w in range(workers)]
    print(f"self-play: {args.games} games, {args.sims} sims/move, {workers} process(es) x {parallel} games at once, "
          f"{'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU'}, model {args.model}", flush=True)

    t0 = time.time()
    if workers == 1:
        results = [worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:   # spawn: CUDA cannot be shared with forked children
            results = pool.map(worker, jobs)
    X, PI, Z, G, wins = [], [], [], [], {0: 0, 1: 0, 2: 0}
    offset = 0
    for x, pi, z, g, st in results:
        X.append(x); PI.append(pi); Z.append(z); G.append(g * workers + offset)
        offset += 1
        for k in wins:
            wins[k] += st[k]
    G = np.concatenate(G)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.concatenate(X), pi=np.concatenate(PI), z=np.concatenate(Z), game=G)
    minutes = (time.time() - t0) / 60
    print(f"saved {len(G)} positions from {args.games} games (X wins {wins[1]}, O wins {wins[2]}, draws {wins[0]}) "
          f"-> {out}  ({minutes:.1f} min, {args.games / max(minutes, 1e-9) * 60:.0f} games/hour)", flush=True)
    return out


if __name__ == "__main__":
    main()
