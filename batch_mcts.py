"""Batched MCTS: many games are searched at the same time and the network sees all their leaves in one batch.

Evaluating one board at a time wastes a GPU (and most of a CPU): the cost is in launching the work,
not in the 15x15 board. Here every active game walks down its own tree until it reaches a position the
network has to judge, then all those positions go through the network together (hundreds at once).
Each game keeps exactly one position waiting at a time, so no virtual loss is needed.

Same search as gomoku.MCTS (PUCT, the same tactics at every node, the same backup), so results match it.

Used by selfplay.py (one model playing itself) and loop.py (gating: new model vs best model).
"""
import time

import numpy as np
import torch

from gomoku import N, GomokuNet, add_root_noise, is_five, make_node, planes, puct_select, tactics

C_PUCT, FPU = 1.5, 0.2


def pick_device(worker=0):
    if torch.cuda.is_available():
        return f"cuda:{worker % torch.cuda.device_count()}"
    return "cpu"


class Evaluator:
    """The network on a device, taking a numpy batch (B, 2, 15, 15) -> (probs (B, 225), values (B,))."""

    def __init__(self, path, device="cpu"):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model = GomokuNet(**ckpt["cfg"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.cuda = device.startswith("cuda")

    @torch.no_grad()
    def __call__(self, x):
        xb = torch.from_numpy(x).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.cuda):
            logits, v = self.model(xb)
        p = logits.float().softmax(1).cpu().numpy()
        v = np.zeros(len(x), np.float32) if v is None else v.float().cpu().numpy()
        return p, v


class Game:
    def __init__(self, gid, board, seed, seats):
        self.id = gid
        self.board = board.copy()
        self.player = 1 if (board == 1).sum() == (board == 2).sum() else 2
        self.seats = seats                  # {1: evaluator index for X, 2: ... for O}
        self.rng = np.random.default_rng(seed)
        self.root = None
        self.need = self.done = 0           # simulations wanted / done for the current move
        self.pending = None                 # (path, board, player, block) waiting for the network
        self.moves, self.X, self.PI, self.who = [], [], [], []
        self.winner = None                  # None while running, 0 draw, 1 / 2 winner


def play_games(tasks, evaluators, sims, parallel=256, noise=None, temp_moves=0, reuse=True,
               record=False, progress=None):
    """tasks: list of (seed, opening board, seats). Plays every game to the end.

    noise:      (alpha, eps) Dirichlet noise at each root (self-play exploration)
    temp_moves: the first moves of each game are sampled from the visit counts instead of the best move
    reuse:      keep the subtree of the move played (only right when one tree serves both sides)
    record:     keep (planes, visit distribution, player) for every move, for training
    progress:   called with each finished Game
    Returns the finished Game objects in task order.
    """
    queue = list(enumerate(tasks))[::-1]
    active, done = [], [None] * len(tasks)

    def set_root(g, node, reused):
        g.root = node
        g.need = max(sims - int(node.N.sum()), sims // 4) if reused else sims
        g.need = max(g.need, 1)
        g.done = 0
        add_root_noise(node, g.board, noise, g.rng)

    def backup(path, leaf):
        v = -leaf                               # value of the last edge for the player who made it
        for nd, a in reversed(path):
            nd.N[a] += 1
            nd.W[a] += v
            v = -v

    def play_move(g):
        root = g.root
        pi = root.N / root.N.sum()
        if record:
            g.X.append(planes(g.board, g.player))
            g.PI.append(pi.astype(np.float16))
            g.who.append(g.player)
        a = int(g.rng.choice(N * N, p=pi)) if len(g.moves) < temp_moves else int(root.N.argmax())
        r, c = divmod(a, N)
        g.board[r, c] = g.player
        g.moves.append(a)
        if is_five(g.board, r, c):
            g.winner = g.player
            return
        if not (g.board == 0).any():
            g.winner = 0
            return
        g.player = 3 - g.player
        child = root.children.get(a) if reuse else None
        g.root = None
        if child is not None:
            set_root(g, child, True)

    def advance(g):
        """Run g's search until it needs the network (sets g.pending) or the game ends."""
        while g.winner is None:
            if g.root is None:
                node, block = tactics(g.board, g.player)
                if node is None:
                    g.pending = ([], g.board.copy(), g.player, block)
                    return
                set_root(g, node, False)
            if g.done >= g.need:
                play_move(g)
                continue
            b, p, node, path = g.board.copy(), g.player, g.root, []
            while True:
                a = puct_select(node, C_PUCT, FPU)
                path.append((node, a))
                r, c = divmod(a, N)
                b[r, c] = p
                if is_five(b, r, c):
                    leaf = -1.0
                    break
                if not (b == 0).any():
                    leaf = 0.0
                    break
                p = 3 - p
                child = node.children.get(a)
                if child is None:
                    child, block = tactics(b, p)
                    if child is None:
                        g.pending = (path, b, p, block)
                        return
                    node.children[a] = child
                    leaf = child.v
                    break
                node = child
            backup(path, leaf)
            g.done += 1

    while queue or active:
        while queue and len(active) < parallel:
            i, (seed, board, seats) = queue.pop()
            active.append(Game(i, board, seed, seats))
        for g in active:
            if g.pending is None:
                advance(g)
        waiting = [g for g in active if g.pending is not None]
        # the tree belongs to the player to move at the root, so that player's network judges its leaves
        by_eval = {}
        for g in waiting:
            by_eval.setdefault(g.seats[g.player], []).append(g)
        for ei, gs in by_eval.items():
            x = np.stack([planes(g.pending[1], g.pending[2]) for g in gs])
            probs, values = evaluators[ei](x)
            for g, p, v in zip(gs, probs, values):
                path, _, _, block = g.pending
                g.pending = None
                node = make_node(p, v, block)
                if not path:
                    set_root(g, node, False)
                else:
                    parent, a = path[-1]
                    parent.children[a] = node
                    backup(path, node.v)
                    g.done += 1
        still = []
        for g in active:
            if g.winner is None:
                still.append(g)
            else:
                done[g.id] = g
                if progress:
                    progress(g)
        active = still
    return done


def match(path_a, path_b, openings, sims, parallel=256, device=None):
    """A vs B from each opening twice (colours swapped). -> A's score in [0, 1], and per-game details."""
    device = device or pick_device()
    evals = [Evaluator(path_a, device), Evaluator(path_b, device)]
    tasks = []
    for k, b in enumerate(openings):
        tasks.append((k, b, {1: 0, 2: 1}))      # A plays X
        tasks.append((k, b, {1: 1, 2: 0}))      # A plays O
    t0 = time.time()
    games = play_games(tasks, evals, sims, parallel=parallel, reuse=False)
    scores = []
    for g, (_, _, seats) in zip(games, tasks):
        a_side = 1 if seats[1] == 0 else 2
        scores.append(0.5 if g.winner == 0 else float(g.winner == a_side))
    s = np.array(scores)
    print(f"match {len(s)} games: A score {s.mean():.3f}  [A as X {s[0::2].mean():.3f}, A as O {s[1::2].mean():.3f}]  "
          f"({time.time() - t0:.0f}s on {device})", flush=True)
    return float(s.mean()), s
