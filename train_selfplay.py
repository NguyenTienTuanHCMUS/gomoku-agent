"""Train the whole network (policy + value) on self-play games, starting from an existing model.

  policy loss: cross-entropy towards pi, the MCTS visit distribution (not just one move)
  value loss:  mean squared error towards z, the final result for the player to move

A slice of the original Gomocup positions (--mix) is added to every epoch so the network does not
forget what it learned from the tournament engines while it adapts to its own games.

    python train_selfplay.py --model models/pv1/gomoku_pv.pt --data data/selfplay/gen1.npz --out models/sp/gen1.pt
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gomoku import GomokuNet, N


def sym(t, k):
    """Same dihedral transform for boards (B, C, 15, 15) and move maps (B, 15, 15)."""
    t = torch.rot90(t, k % 4, dims=(-2, -1))
    return torch.flip(t, dims=(-1,)) if k >= 4 else t


def load_selfplay(paths, val_every):
    X, PI, Z, VAL = [], [], [], []
    for f, path in enumerate(paths):
        d = np.load(path)
        X.append(d["X"]); PI.append(d["pi"]); Z.append(d["z"])
        VAL.append((d["game"] + f) % val_every == 0)       # whole games go to validation
    return (np.concatenate(X), np.concatenate(PI).astype(np.float32),
            np.concatenate(Z).astype(np.float32), np.concatenate(VAL))


def load_gomocup(path, count, val_count, seed):
    """Random train positions (one-hot target = the engine's move) and held-out val positions,
    using the same game-level split as train_policy.ipynb (rng seed 0, 10% of games)."""
    d = np.load(path)
    X, y, z, game = d["X"], d["y"].astype(np.int64), d["z"].astype(np.float32), d["game"]
    rng = np.random.default_rng(0)
    val_games = rng.choice(np.unique(game), size=len(np.unique(game)) // 10, replace=False)
    is_val = np.isin(game, val_games)
    rng = np.random.default_rng(seed)
    tr = rng.choice(np.where(~is_val)[0], size=min(count, int((~is_val).sum())), replace=False)
    va = rng.choice(np.where(is_val)[0], size=val_count, replace=False)
    pi = np.zeros((len(tr), N * N), np.float32)
    pi[np.arange(len(tr)), y[tr]] = 1
    return X[tr], pi, z[tr], X[va], y[va]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pv1/gomoku_pv.pt", help="starting weights")
    ap.add_argument("--data", nargs="+", required=True, help="self-play .npz files")
    ap.add_argument("--out", default="models/sp/gen1.pt")
    ap.add_argument("--gomocup", default="data/gomoku_15x15.npz")
    ap.add_argument("--mix", type=float, default=0.25, help="Gomocup positions added, as a fraction of self-play positions")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=20, help="every n-th self-play game is held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = {**ckpt["cfg"], "value": True}
    model = GomokuNet(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected and all(k.startswith("value_") for k in missing), (missing, unexpected)
    model.to(dev)

    X, PI, Z, VAL = load_selfplay(args.data, args.val_every)
    tr, va = np.where(~VAL)[0], np.where(VAL)[0]
    gX, gPI, gZ, gvX, gvy = load_gomocup(args.gomocup, int(len(tr) * args.mix), 4000, args.seed)
    # training set = self-play train positions + Gomocup slice; Gomocup results that are unknown (z = 0) get no value loss
    tX = torch.from_numpy(np.concatenate([X[tr], gX]))
    tPI = torch.from_numpy(np.concatenate([PI[tr], gPI]))
    tZ = torch.from_numpy(np.concatenate([Z[tr], gZ]))
    tW = torch.from_numpy(np.concatenate([np.ones(len(tr), np.float32), (gZ != 0).astype(np.float32)]))
    print(f"self-play: {len(tr)} train / {len(va)} val positions from {len(args.data)} file(s); "
          f"+ {len(gX)} Gomocup positions; device {dev}", flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        ce, agree, mse = [], [], []
        for s in range(0, len(va), 1024):
            b = va[s:s + 1024]
            logits, v = model(torch.from_numpy(X[b]).to(dev))
            pi = torch.from_numpy(PI[b]).to(dev)
            ce.append(-(pi * logits.log_softmax(1)).sum(1).cpu())
            agree.append((logits.argmax(1) == pi.argmax(1)).float().cpu())
            mse.append(((v - torch.from_numpy(Z[b]).to(dev)) ** 2).cpu())
        top1 = []
        for s in range(0, len(gvX), 1024):
            logits, _ = model(torch.from_numpy(gvX[s:s + 1024]).to(dev))
            top1.append((logits.argmax(1).cpu() == torch.from_numpy(gvy[s:s + 1024])).float())
        return (float(torch.cat(ce).mean()), float(torch.cat(agree).mean()),
                float(torch.cat(mse).mean()), float(torch.cat(top1).mean()))

    def report(tag, r):
        print(f"{tag}  self-play val: policy loss {r[0]:.3f}, picks MCTS's move {r[1]:.3f}, value mse {r[2]:.3f}  |  "
              f"Gomocup val top-1 {r[3]:.3f}", flush=True)

    report("before     ", evaluate())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(tX) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tX))
        t0, sums, n = time.time(), np.zeros(2), 0
        for s in range(0, len(perm), args.bs):
            b = perm[s:s + args.bs]
            k = int(torch.randint(8, ()))
            xb = sym(tX[b], k).to(dev)
            pib = sym(tPI[b].view(-1, N, N), k).reshape(-1, N * N).to(dev)
            logits, v = model(xb)
            lp = -(pib * logits.log_softmax(1)).sum(1).mean()
            w = tW[b].to(dev)
            lv = (w * (v - tZ[b].to(dev)) ** 2).sum() / w.sum().clamp(min=1)
            loss = lp + args.value_weight * lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            sums += (lp.item() * len(b), lv.item() * len(b))
            n += len(b)
        report(f"epoch {ep + 1}/{args.epochs}", evaluate())
        print(f"           train policy loss {sums[0] / n:.3f}, value mse {sums[1] / n:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, out)
    print(f"-> {out}")
    return out


if __name__ == "__main__":
    main()
