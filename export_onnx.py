"""Export a checkpoint to ONNX for the web app (docs/), and check it gives the same numbers as PyTorch.

    python -m pip install onnx onnxscript onnxruntime
    python export_onnx.py --model models/pv2/gomoku_pv.pt --out docs/model.onnx
"""
import argparse

import numpy as np
import torch

from gomoku import load_model


class Export(torch.nn.Module):
    """float32 board (B, 2, 15, 15) [to move, opponent] -> policy logits (B, 225) with occupied = -1e9, value (B,)"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, board):
        logits, v = self.model(board)
        return logits, (v if v is not None else torch.zeros_like(logits[:, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pv2/gomoku_pv.pt")
    ap.add_argument("--out", default="docs/model.onnx")
    args = ap.parse_args()

    net = Export(load_model(args.model)).eval()
    dummy = torch.zeros(1, 2, 15, 15)
    torch.onnx.export(net, (dummy,), args.out, input_names=["board"], output_names=["policy", "value"],
                      dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out)
    d = np.load("data/gomoku_15x15.npz")
    x = d["X"][np.random.default_rng(0).choice(len(d["y"]), 64, replace=False)].astype(np.float32)
    with torch.no_grad():
        pt_logits, pt_v = net(torch.from_numpy(x))
    ox_logits, ox_v = sess.run(None, {"board": x})
    legal = x.sum(1).reshape(len(x), -1) == 0
    print(f"max |diff| policy {np.abs(pt_logits.numpy() - ox_logits)[legal].max():.2e}, "
          f"value {np.abs(pt_v.numpy() - ox_v).max():.2e}  -> {args.out}")


if __name__ == "__main__":
    main()
