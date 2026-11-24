// Gomoku engine running in a Web Worker: the ONNX network + MCTS, a port of gomoku.py.
// Board: Int8Array(225), 0 empty, 1 = X (moves first), 2 = O. Moves are flat indices row * 15 + col.
/* global ort */
importScripts('vendor/ort/ort.wasm.min.js');
ort.env.wasm.wasmPaths = new URL('vendor/ort/', self.location.href).href;
ort.env.wasm.numThreads = 1;

const N = 15, NN = 225, BATCH = 8, C_PUCT = 1.5, FPU = 0.2;
let session = null;

// ------------------------------------------------------------ board helpers
const WINDOWS = (() => {
  const w = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++)
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      const rr = r + 4 * dr, cc = c + 4 * dc;
      if (rr >= 0 && rr < N && cc >= 0 && cc < N) for (let k = 0; k < 5; k++) w.push((r + k * dr) * N + c + k * dc);
    }
  return Int16Array.from(w);
})();

function winCells(b, p) {           // empty cells where p would make five or more
  const out = new Set();
  for (let i = 0; i < WINDOWS.length; i += 5) {
    let own = 0, empty = -1, ok = true;
    for (let k = 0; k < 5; k++) {
      const v = b[WINDOWS[i + k]];
      if (v === p) own++;
      else if (v === 0 && empty < 0) empty = WINDOWS[i + k];
      else { ok = false; break; }
    }
    if (ok && own === 4 && empty >= 0) out.add(empty);
  }
  return [...out];
}

function isFive(b, idx) {
  const p = b[idx], r = Math.floor(idx / N), c = idx % N;
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    let n = 1;
    for (const s of [1, -1]) {
      let rr = r + s * dr, cc = c + s * dc;
      while (rr >= 0 && rr < N && cc >= 0 && cc < N && b[rr * N + cc] === p) { n++; rr += s * dr; cc += s * dc; }
    }
    if (n >= 5) return true;
  }
  return false;
}

function only(cells) {
  const P = new Float32Array(NN);
  for (const i of cells) P[i] = 1 / cells.length;
  return P;
}

// ------------------------------------------------------------ network
async function evaluateBatch(items) {      // items: [{board, player}] -> [{P, v}]
  const x = new Float32Array(items.length * 2 * NN);
  items.forEach(({ board, player }, j) => {
    const o = j * 2 * NN;
    for (let i = 0; i < NN; i++) {
      if (board[i] === player) x[o + i] = 1;
      else if (board[i] === 3 - player) x[o + NN + i] = 1;
    }
  });
  const out = await session.run({ board: new ort.Tensor('float32', x, [items.length, 2, N, N]) });
  const logits = out.policy.data, values = out.value.data;
  return items.map((_, j) => {
    const P = new Float32Array(NN);
    let mx = -Infinity, sum = 0;
    for (let i = 0; i < NN; i++) mx = Math.max(mx, logits[j * NN + i]);
    for (let i = 0; i < NN; i++) { P[i] = Math.exp(logits[j * NN + i] - mx); sum += P[i]; }
    for (let i = 0; i < NN; i++) P[i] /= sum;
    return { P, v: values[j] };
  });
}

// ------------------------------------------------------------ MCTS
class Node {
  constructor(P, v) { this.P = P; this.v = v; this.N = new Float32Array(NN); this.W = new Float32Array(NN); this.total = 0; this.children = new Map(); }
  select() {
    const sq = Math.sqrt(this.total + 1);
    let best = -1, bestScore = -Infinity;
    for (let a = 0; a < NN; a++) {
      if (this.P[a] <= 0) continue;
      const n = this.N[a];
      const s = (n > 0 ? this.W[a] / n : this.v - FPU) + C_PUCT * this.P[a] * sq / (1 + n);
      if (s > bestScore) { bestScore = s; best = a; }
    }
    return best;
  }
}

// Exact tactics before the network: win now / opponent double threat / forced block.
function tactics(board, player) {
  const mine = winCells(board, player);
  if (mine.length) return { node: new Node(only(mine), 1) };
  const theirs = winCells(board, 3 - player);
  if (theirs.length >= 2) return { node: new Node(only(theirs), -1) };
  return { forced: theirs.length === 1 ? theirs : null };
}

async function expandOne(board, player) {
  const t = tactics(board, player);
  if (t.node) return t.node;
  const [{ P, v }] = await evaluateBatch([{ board, player }]);
  return new Node(t.forced ? only(t.forced) : P, v);
}

// Each simulation adds a "virtual loss" (N += 1, W -= 1) on its path so the next simulations in the
// same batch explore elsewhere; backup then adds v + 1 to undo it and record the real value.
function backup(path, leafValue) {
  let v = -leafValue;
  for (let i = path.length - 1; i >= 0; i--) {
    const [node, a] = path[i];
    node.W[a] += v + 1;
    v = -v;
  }
}

// Tree reuse: if the board is the last searched position plus our chosen move plus one reply,
// continue from that grandchild so the visits already spent there are not thrown away.
let previous = null;   // { board, player, root }
function reusableRoot(board, player) {
  if (!previous || previous.player !== player) return null;
  const added = [];
  for (let i = 0; i < NN; i++) if (previous.board[i] !== board[i]) {
    if (previous.board[i] !== 0) return null;
    added.push(i);
  }
  const mine = added.filter(i => board[i] === player), theirs = added.filter(i => board[i] === 3 - player);
  if (added.length !== 2 || mine.length !== 1 || theirs.length !== 1) return null;
  const child = previous.root.children.get(mine[0]);
  return (child && child.children.get(theirs[0])) || null;
}

async function search(board, player, sims, onProgress) {
  const root = reusableRoot(board, player) || await expandOne(board, player);
  previous = { board: board.slice(), player, root };
  let done = 0, lastReport = 0;
  while (done < sims) {
    const pending = [], pendingEdges = new Map();
    while (pending.length < BATCH && done + pending.length < sims) {
      const b = board.slice();
      let p = player, node = root, leaf = null;
      const path = [];
      for (;;) {
        const a = node.select();
        path.push([node, a]);
        node.N[a] += 1; node.W[a] -= 1; node.total += 1;
        b[a] = p;
        if (isFive(b, a)) { leaf = { value: -1 }; break; }
        if (!b.includes(0)) { leaf = { value: 0 }; break; }
        p = 3 - p;
        const child = node.children.get(a);
        if (child) { node = child; continue; }
        const t = tactics(b, p);
        if (t.node) { node.children.set(a, t.node); leaf = { value: t.node.v }; break; }
        let edges = pendingEdges.get(node);
        if (edges && edges.has(a)) { leaf = { collision: true }; break; }
        if (!edges) pendingEdges.set(node, (edges = new Set()));
        edges.add(a);
        leaf = { eval: { board: b, player: p, parent: node, a, forced: t.forced, path } };
        break;
      }
      if (leaf.collision) {                     // same unexpanded leaf twice: undo this path, run the batch
        for (const [nd, a] of path) { nd.N[a] -= 1; nd.W[a] += 1; nd.total -= 1; }
        break;
      }
      if (leaf.eval) pending.push(leaf.eval);
      else { backup(path, leaf.value); done++; }
    }
    if (pending.length) {
      const res = await evaluateBatch(pending);
      pending.forEach((q, j) => {
        const node = new Node(q.forced ? only(q.forced) : res[j].P, res[j].v);
        q.parent.children.set(q.a, node);
        backup(q.path, node.v);
      });
      done += pending.length;
    }
    if (onProgress && done - lastReport >= 32) { lastReport = done; onProgress(done); }
  }
  return root;
}

async function policyMove(board, player) {
  const [{ P, v }] = await evaluateBatch([{ board, player }]);
  for (const who of [player, 3 - player]) {
    const cells = winCells(board, who);
    if (cells.length) return { move: cells.reduce((a, b) => (P[b] > P[a] ? b : a)), P, v };
  }
  let best = 0;
  for (let i = 1; i < NN; i++) if (P[i] > P[best]) best = i;
  return { move: best, P, v };
}

function topK(arr, k) {
  return [...arr.keys()].filter(i => arr[i] > 0).sort((a, b) => arr[b] - arr[a]).slice(0, k);
}

// ------------------------------------------------------------ messages
self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === 'init') {
      session = await ort.InferenceSession.create(e.data.model, { executionProviders: ['wasm'] });
      const t = performance.now();
      await evaluateBatch([{ board: new Int8Array(NN), player: 1 }]);
      self.postMessage({ id, ok: true, ms: performance.now() - t });
    } else if (type === 'move') {
      const board = Int8Array.from(e.data.board), player = e.data.player, sims = e.data.sims;
      const t = performance.now();
      if (!board.includes(1) && !board.includes(2)) {
        self.postMessage({ id, move: 112, reason: 'center', value: 0, top: [], ms: 0 });
        return;
      }
      if (sims <= 0) {
        const { move, P, v } = await policyMove(board, player);
        self.postMessage({ id, move, reason: 'policy', value: v, top: topK(P, 3).map(i => [i, P[i]]), ms: performance.now() - t });
        return;
      }
      const root = await search(board, player, sims, done => self.postMessage({ id, progress: done }));
      const move = topK(root.N, 1)[0];
      const total = root.N.reduce((a, b) => a + b, 0);
      self.postMessage({
        id, move, reason: 'mcts', sims: total, ms: performance.now() - t,
        value: root.W[move] / Math.max(1, root.N[move]),
        top: topK(root.N, 3).map(i => [i, root.N[i] / total]),
      });
    } else if (type === 'hint') {
      const board = Int8Array.from(e.data.board);
      const [{ P, v }] = await evaluateBatch([{ board, player: e.data.player }]);
      self.postMessage({ id, value: v, top: topK(P, 5).map(i => [i, P[i]]) });
    }
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
