"""The self-play loop, AlphaZero style, starting from an already trained model.

One generation:
  1. self-play:  the current best model plays --games games against itself  -> data/selfplay/genG.npz
  2. train:      a copy of the best model learns from the last --window generations of games -> models/sp/genG.pt
  3. gate:       the new model plays the best model (arena.py); if it scores at least --threshold
                 it becomes the new best, otherwise it is thrown away and the next generation tries again
                 (with more games in the window).

Progress is kept in runs/selfplay/state.json, so the loop can be stopped (Ctrl+C) and started again:
finished steps are skipped.

    python loop.py --gens 5                     # 5 generations from models/pv2/gomoku_pv.pt
    python loop.py --gens 3 --games 400         # smaller generations
    python loop.py --gens 50 --hours 11         # as many as fit in 11 hours (Kaggle sessions stop at 12)

Everything uses the GPU when PyTorch sees one, else the CPU.
"""
import argparse
import json
import time
from pathlib import Path

import arena
import batch_mcts
import selfplay
import train_selfplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="models/pv2/gomoku_pv.pt", help="first best model")
    ap.add_argument("--gens", type=int, default=5, help="generations to run in this call")
    ap.add_argument("--games", type=int, default=800, help="self-play games per generation")
    ap.add_argument("--sims", type=int, default=200, help="MCTS simulations per move in self-play")
    ap.add_argument("--window", type=int, default=3, help="train on the games of the last N generations")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--gate-openings", type=int, default=40, help="each opening is played twice, colours swapped")
    ap.add_argument("--gate-sims", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--hours", type=float, default=0, help="stop before a generation that would end past this (0 = no limit)")
    ap.add_argument("--workers", type=int, default=0, help="self-play processes (0 = auto)")
    ap.add_argument("--state", default="runs/selfplay/state.json")
    args = ap.parse_args()

    state_path = Path(args.state)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"best": args.start, "next_gen": 1, "history": []}

    def save():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    start = time.time()
    for _ in range(args.gens):
        g = state["next_gen"]
        t0 = time.time()
        recent = [h["minutes"] for h in state["history"][-3:]]
        if args.hours and recent and (t0 - start) / 60 + max(recent) > args.hours * 60:
            print(f"\nstopping: another generation (~{max(recent):.0f} min) would pass the {args.hours} h limit")
            break
        print(f"\n===== generation {g}  (best so far: {state['best']}) =====", flush=True)

        data = Path(f"data/selfplay/gen{g}.npz")
        if not data.exists():
            selfplay.main(["--model", state["best"], "--games", str(args.games), "--sims", str(args.sims),
                           "--out", str(data), "--seed", str(g), "--workers", str(args.workers)])

        cand = Path(f"models/sp/gen{g}.pt")
        if not cand.exists():
            window = [f"data/selfplay/gen{i}.npz" for i in range(max(1, g - args.window + 1), g + 1)
                      if Path(f"data/selfplay/gen{i}.npz").exists()]
            train_selfplay.main(["--model", state["best"], "--data", *window, "--out", str(cand),
                                 "--epochs", str(args.epochs), "--seed", str(g)])

        print(f"\ngate: gen{g} vs {state['best']}", flush=True)
        ops = arena.openings("data/gomoku_15x15.npz", args.gate_openings, 1000 + g)
        score, _ = batch_mcts.match(str(cand), state["best"], ops, args.gate_sims)
        promoted = score >= args.threshold
        state["history"].append({"gen": g, "vs": state["best"], "score": round(score, 3), "promoted": promoted,
                                 "minutes": round((time.time() - t0) / 60, 1)})
        if promoted:
            state["best"] = str(cand).replace("\\", "/")
        state["next_gen"] = g + 1
        save()
        print(f"generation {g}: score {score:.3f} vs previous best -> "
              f"{'NEW BEST' if promoted else 'kept the old best'}  ({(time.time() - t0) / 60:.0f} min)", flush=True)

    print("\nhistory:")
    for h in state["history"]:
        print(f"  gen {h['gen']}: {h['score']:.3f} vs {h['vs']}  {'promoted' if h['promoted'] else ''}")
    print(f"best model: {state['best']}")


if __name__ == "__main__":
    main()
