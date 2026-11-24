"""Train a value head on top of the trained policy network, keeping the policy part frozen.

The value head predicts the game result for the player to move (+1 win, -1 loss) from the
features the policy network already computes. Only ~6k new parameters are trained, so a CPU is enough.

    python train_value.py                    # -> models/pv1/gomoku_pv.pt (policy + value)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gomoku import GomokuNet, N


def sym(t, k):
    t = torch.rot90(t, k % 4, dims=(-2, -1))
    return torch.flip(t, dims=(-1,)) if k >= 4 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="models/run1/gomoku_policy.pt")
    ap.add_argument("--data", default="data/gomoku_15x15.npz")
    ap.add_argument("--out", default="models/pv1/gomoku_pv.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--per-epoch", type=int, default=0, help="samples per epoch (0 = all)")
    args = ap.parse_args()
    torch.manual_seed(0)

    ckpt = torch.load(args.policy, map_location="cpu", weights_only=True)
    cfg = {**ckpt["cfg"], "value": True}
    model = GomokuNet(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected and all(k.startswith("value_") for k in missing), (missing, unexpected)
    for p in model.parameters():
        p.requires_grad = False
    vparams = [p for n, p in model.named_parameters() if n.startswith("value_")]
    for p in vparams:
        p.requires_grad = True
    print(f"value head: {sum(p.numel() for p in vparams)} trainable params, policy frozen")

    d = np.load(args.data)
    X, z, game = d["X"], d["z"].astype(np.float32), d["game"]
    # same game-level split as train_policy.ipynb, so val games were never seen by the policy either
    rng = np.random.default_rng(0)
    val_games = rng.choice(np.unique(game), size=len(np.unique(game)) // 10, replace=False)
    is_val = np.isin(game, val_games)
    known = z != 0
    tr_idx, va_idx = np.where(~is_val & known)[0], np.where(is_val & known)[0]
    stones = X.sum((1, 2, 3))
    print(f"train {len(tr_idx)}  val {len(va_idx)}")
    Xt, zt = torch.from_numpy(X), torch.from_numpy(z)

    opt = torch.optim.AdamW(vparams, lr=args.lr, weight_decay=1e-4)
    per_epoch = args.per_epoch or len(tr_idx)
    steps = args.epochs * ((per_epoch + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    @torch.no_grad()
    def evaluate():
        model.eval()
        preds = []
        for s in range(0, len(va_idx), 2048):
            b = torch.from_numpy(va_idx[s:s + 2048])
            preds.append(model.value_from(model.trunk(Xt[b])))
        v = torch.cat(preds).numpy()
        t = z[va_idx]
        mse = float(((v - t) ** 2).mean())
        acc = float((np.sign(v) == t).mean())
        phase = {}
        for name, lo, hi in (("<15 quân", 0, 15), ("15-29", 15, 30), ("30+", 30, 999)):
            m = (stones[va_idx] >= lo) & (stones[va_idx] < hi)
            phase[name] = float((np.sign(v[m]) == t[m]).mean())
        return mse, acc, phase

    mse, acc, phase = evaluate()
    print(f"before: val mse {mse:.3f} (always-0 guess = 1.000)  sign acc {acc:.3f}")
    best = 9.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.eval()
        model.value_conv.train()   # trunk BatchNorm stays frozen, value BatchNorm learns
        perm = torch.from_numpy(np.random.permutation(tr_idx)[:per_epoch])
        t0, tl, tn = time.time(), 0.0, 0
        for s in range(0, len(perm), args.bs):
            b = perm[s:s + args.bs]
            k = int(torch.randint(8, ()))
            xb = sym(Xt[b], k)
            with torch.no_grad():
                h = model.trunk(xb)
            loss = F.mse_loss(model.value_from(h), zt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
            tn += len(b)
        mse, acc, phase = evaluate()
        ph = "  ".join(f"{k}: {v:.3f}" for k, v in phase.items())
        print(f"ep {ep + 1}/{args.epochs}  train mse {tl / tn:.3f}  val mse {mse:.3f}  sign acc {acc:.3f}  [{ph}]  ({time.time() - t0:.0f}s)")
        if mse < best:
            best = mse
            torch.save({"cfg": cfg, "state_dict": model.state_dict()}, out)
    print(f"best val mse {best:.3f} -> {out}")


if __name__ == "__main__":
    main()
