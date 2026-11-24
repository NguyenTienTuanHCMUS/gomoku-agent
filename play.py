"""Play Gomoku 15x15 against the trained network in the terminal.

    python play.py                  # you play X (first move), the model searches 400 MCTS simulations per move
    python play.py --ai-first       # the model plays X
    python play.py --sims 1000      # stronger and slower; --sims 0 = no search (network + win/block check)
    python play.py --model models/run1/gomoku_policy.pt   # a policy-only checkpoint also works

Enter moves as column letter + row number, e.g. h8 (the centre).
Commands: hint (model's top 5 for you), undo, quit.
"""
import argparse
import os
import re
import sys
import time

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    sys.exit("Cần cài PyTorch: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")

from gomoku import MCTS, N, evaluate, is_five, load_model, policy_move

COLS = "abcdefghijklmno"
SYM = {0: ".", 1: "X", 2: "O"}
DEFAULT_MODEL = "models/pv1/gomoku_pv.pt" if os.path.exists("models/pv1/gomoku_pv.pt") else "models/run1/gomoku_policy.pt"


def name(move):
    r, c = move
    return f"{COLS[c]}{r + 1}"


def parse(text):
    m = re.fullmatch(r"([a-o])\s*(\d{1,2})", text)
    if not m or not 1 <= int(m[2]) <= N:
        return None
    return int(m[2]) - 1, COLS.index(m[1])


def show(board, last):
    print("\n    " + "  ".join(COLS))
    for r in range(N):
        cells = []
        for c in range(N):
            s = SYM[int(board[r, c])]
            cells.append(f"[{s}]" if (r, c) == last else f" {s} ")
        print(f"{r + 1:>2} " + "".join(cells))
    print()


def top_moves(p, k=5):
    idx = np.argsort(p)[::-1][:k]
    return ", ".join(f"{name(divmod(int(i), N))} {p[i]:.0%}" for i in idx)


def win_pct(v):
    return f"{(v + 1) / 2:.0%}"


def ai_turn(model, search, board, player):
    """Pick the model's move and describe why."""
    if search is None or not board.any():
        move, reason, p, v = policy_move(model, board, player)
        msg = f"{reason}, mạng cho nước này {p[move[0] * N + move[1]]:.0%}. Top 3: {top_moves(p, 3)}"
        if v is not None:
            msg += f". Model tự đánh giá khả năng thắng: {win_pct(v)}"
        return move, msg
    t = time.time()
    move, root = search.best_move(board, player)
    a = move[0] * N + move[1]
    visits = root.N.sum()
    top = np.argsort(-root.N)[:3]
    alts = ", ".join(f"{name(divmod(int(i), N))} {root.N[i] / visits:.0%}" for i in top if root.N[i] > 0)
    q = root.W[a] / max(root.N[a], 1)
    return move, (f"MCTS {int(visits)} lượt trong {time.time() - t:.1f}s; nước được thử nhiều nhất: {alts}. "
                  f"Khả năng thắng ước tính sau nước này: {win_pct(q)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sims", type=int, default=400, help="MCTS simulations per move (0 = no search)")
    ap.add_argument("--ai-first", action="store_true", help="model plays X and moves first")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    model = load_model(args.model)
    search = MCTS(model, sims=args.sims) if args.sims > 0 else None
    board = np.zeros((N, N), np.int8)
    history = []
    human = 2 if args.ai_first else 1
    player = 1
    print(f"Model: {args.model} ({'có' if model.has_value else 'không có'} value head), "
          f"{'MCTS ' + str(args.sims) + ' lượt/nước' if search else 'không tìm kiếm'}.")
    print(f"Bạn cầm {SYM[human]}, model cầm {SYM[3 - human]}. X đi trước. "
          "Nhập nước kiểu h8; lệnh: hint, undo, quit.")
    show(board, None)

    while True:
        if not (board == 0).any():
            print("Hoà: bàn cờ đã đầy.")
            break

        if player == human:
            try:
                text = input(f"Bạn ({SYM[human]}) đi: ").strip().lstrip("﻿").lower()
            except EOFError:
                break
            if text in ("q", "quit", "exit"):
                break
            if text == "hint":
                p, v = evaluate(model, board, human)
                print("Model gợi ý:", top_moves(p) + (f". Khả năng thắng của bạn theo model: {win_pct(v)}" if v is not None else ""))
                continue
            if text == "undo":
                if len(history) < 2:
                    print("Chưa có nước nào của bạn để đi lại.")
                    continue
                for _ in range(2):
                    board[history.pop()] = 0
                show(board, history[-1] if history else None)
                continue
            move = parse(text)
            if move is None or board[move]:
                print("Nước không hợp lệ: nhập chữ cột a-o và số hàng 1-15, vào ô còn trống (ví dụ h8).")
                continue
            msg = None
        else:
            move, msg = ai_turn(model, search, board, player)

        board[move] = player
        history.append(move)
        show(board, move)
        if msg:
            print(f"Model ({SYM[player]}) đi {name(move)}: {msg}")
        if is_five(board, *move):
            print("Bạn thắng!" if player == human else "Model thắng!")
            break
        player = 3 - player


if __name__ == "__main__":
    main()
