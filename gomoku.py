"""Shared pieces: the network (policy + optional value head), board helpers and MCTS.

Board convention everywhere: numpy int8 (15, 15), 0 empty, 1 = X (moves first), 2 = O.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 15


# ---------------------------------------------------------------- network

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(h)))


class GomokuNet(nn.Module):
    """Policy net from train_policy.ipynb (same parameter names, so its checkpoints load as-is),
    plus an optional value head: tanh output in [-1, 1] = expected result for the player to move."""

    def __init__(self, channels=96, blocks=8, value=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 5, padding=2, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(32, 1, 1))
        if value:
            self.value_conv = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
            self.value_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.has_value = value

    def trunk(self, board):
        x = board.float()
        x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
        return self.body(self.stem(x))

    def policy_from(self, h, board):
        logits = self.head(h).flatten(1).float()
        occupied = (board[:, 0] + board[:, 1]).flatten(1) > 0
        return logits.masked_fill(occupied, -1e9)

    def value_from(self, h):
        v = self.value_conv(h).flatten(2)                       # (B, 32, 225)
        v = torch.cat([v.mean(2), v.amax(2)], 1)                # mean + max pooling -> (B, 64)
        return self.value_fc(v.float()).squeeze(1)

    def forward(self, board):
        """board: uint8 (B, 2, 15, 15) [player to move, opponent] -> (masked policy logits (B, 225), value (B,) or None)"""
        h = self.trunk(board)
        return self.policy_from(h, board), (self.value_from(h) if self.has_value else None)


def load_model(path, threads=2):
    # one 15x15 board at a time: a few threads are faster than all cores fighting over tiny convolutions
    torch.set_num_threads(threads)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(ckpt["cfg"])
    model = GomokuNet(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def planes(board, player):
    return np.stack([board == player, board == 3 - player]).astype(np.uint8)


@torch.no_grad()
def evaluate(model, board, player):
    """-> (move probabilities (225,), value for `player` in [-1, 1] or None)"""
    logits, v = model(torch.from_numpy(planes(board, player))[None])
    return logits[0].softmax(0).numpy(), (None if v is None else float(v[0]))


# ---------------------------------------------------------------- board helpers

def _windows():
    w = []
    for r in range(N):
        for c in range(N):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc = r + 4 * dr, c + 4 * dc
                if 0 <= rr < N and 0 <= cc < N:
                    w.append([(r + k * dr) * N + c + k * dc for k in range(5)])
    return np.array(w)


WINDOWS = _windows()  # (572, 5): flat indices of every 5-in-a-row segment


def win_cells(board, player):
    """Flat indices of empty cells where `player` would make five (or more) in a row."""
    flat = board.ravel()
    own = (flat == player)[WINDOWS].sum(1)
    emp = (flat == 0)[WINDOWS].sum(1)
    w = WINDOWS[(own == 4) & (emp == 1)]
    return np.unique(w[flat[w] == 0])


def is_five(board, r, c):
    p = board[r, c]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < N and 0 <= cc < N and board[rr, cc] == p:
                n += 1
                rr += s * dr
                cc += s * dc
        if n >= 5:
            return True
    return False


def policy_move(model, board, player):
    """No search: win now if possible, else block a five-threat, else the network's top move.
    -> ((row, col), reason, probs, value)"""
    p, v = evaluate(model, board, player)
    if not board.any():
        return (N // 2, N // 2), "khai cuộc ở giữa bàn", p, v
    for who, reason in ((player, "thắng ngay"), (3 - player, "chặn nước thắng")):
        cells = win_cells(board, who)
        if len(cells):
            return divmod(int(cells[p[cells].argmax()]), N), reason, p, v
    return divmod(int(p.argmax()), N), "mạng nơ-ron", p, v


# ---------------------------------------------------------------- MCTS

class Node:
    __slots__ = ("P", "N", "W", "v", "children")

    def __init__(self, priors, v):
        self.P = priors                       # (225,) prior probabilities, 0 for moves not considered
        self.N = np.zeros(N * N, np.float32)  # visit counts per move
        self.W = np.zeros(N * N, np.float32)  # summed values per move, from this node's player's view
        self.v = v                            # value estimate of this position for the player to move
        self.children = {}


class MCTS:
    """AlphaZero-style PUCT search guided by the network.

    Exact tactics are applied at every node before asking the network:
      - the player to move can make five       -> value +1, only winning moves considered
      - the opponent has two or more five-threats -> value -1 (cannot block both)
      - the opponent has one five-threat        -> only the blocking move considered
    """

    def __init__(self, model, sims=400, c_puct=1.5, fpu=0.2):
        self.model, self.sims, self.c_puct, self.fpu = model, sims, c_puct, fpu

    def expand(self, board, player):
        mine = win_cells(board, player)
        if len(mine):
            return Node(_only(mine), 1.0)
        theirs = win_cells(board, 3 - player)
        if len(theirs) >= 2:
            return Node(_only(theirs), -1.0)
        p, v = evaluate(self.model, board, player)
        if len(theirs) == 1:
            p = _only(theirs)
        return Node(p, 0.0 if v is None else v)

    def select(self, node):
        n = node.N
        q = np.where(n > 0, node.W / np.maximum(n, 1), node.v - self.fpu)
        u = self.c_puct * node.P * math.sqrt(n.sum() + 1) / (1 + n)
        return int(np.where(node.P > 0, q + u, -np.inf).argmax())

    def search(self, board, player):
        """Run the simulations; returns the root (root.N = visit counts, root.W / root.N = values)."""
        root = self.expand(board, player)
        for _ in range(self.sims):
            b, p, node, path = board.copy(), player, root, []
            while True:
                a = self.select(node)
                path.append((node, a))
                r, c = divmod(a, N)
                b[r, c] = p
                if is_five(b, r, c):
                    leaf = -1.0           # the player now to move has lost
                    break
                if not (b == 0).any():
                    leaf = 0.0            # board full: draw
                    break
                p = 3 - p
                child = node.children.get(a)
                if child is None:
                    child = node.children[a] = self.expand(b, p)
                    leaf = child.v
                    break
                node = child
            v = -leaf                     # value of the last edge for the player who made it
            for nd, a in reversed(path):
                nd.N[a] += 1
                nd.W[a] += v
                v = -v
        return root

    def best_move(self, board, player):
        """-> ((row, col), root)"""
        if not board.any():
            return (N // 2, N // 2), None
        root = self.search(board, player)
        return divmod(int(root.N.argmax()), N), root


def _only(cells):
    p = np.zeros(N * N, np.float32)
    p[cells] = 1.0 / len(cells)
    return p
