/**
 * PUCT Monte-Carlo Tree Search, as in AlphaZero.
 * Edge statistics live in the parent node; values are stored from the perspective
 * of the player choosing at that node.
 */
import { CELLS, Game, goalRow, rowOf } from '../engine/quoridor.ts';
import type { Evaluator } from './evaluator.ts';

export interface MctsOptions {
  cPuct: number;
  fpuReduction: number;
  /** Dirichlet noise at the root (self-play only) */
  dirichletAlpha: number;
  dirichletEps: number;
}

export const DEFAULT_MCTS: MctsOptions = {
  cPuct: 1.6,
  fpuReduction: 0.25,
  dirichletAlpha: 0,
  dirichletEps: 0,
};

export class Node {
  actions: Int16Array;
  priors: Float32Array;
  N: Int32Array;
  W: Float32Array;
  children: (Node | null)[];
  visits = 0;
  /** evaluator's value for the side to move at this node */
  value: number;
  /** terminal / proven result for side to move (+1 win, -1 loss) or null */
  proven: number | null = null;

  constructor(actions: number[], priors: Float32Array, value: number) {
    this.actions = Int16Array.from(actions);
    this.priors = priors;
    this.value = value;
    const n = actions.length;
    this.N = new Int32Array(n);
    this.W = new Float32Array(n);
    this.children = new Array(n).fill(null);
  }
}

export interface MoveStat {
  action: number;
  visits: number;
  /** mean value for the side to move at the root, in [-1, 1] */
  q: number;
  prior: number;
  /** principal variation following this move */
  pv: number[];
}

export interface SearchResult {
  /** root value estimate for side to move */
  value: number;
  visits: number;
  moves: MoveStat[]; // sorted by visits desc
}

export class MCTS {
  root: Node | null = null;
  rootGame: Game;
  evaluator: Evaluator;
  opts: MctsOptions;
  rng: () => number;

  constructor(game: Game, evaluator: Evaluator, opts: Partial<MctsOptions> = {}, rng: () => number = Math.random) {
    this.rootGame = game.clone();
    this.evaluator = evaluator;
    this.opts = { ...DEFAULT_MCTS, ...opts };
    this.rng = rng;
  }

  /** Expand a node for a non-terminal game state. */
  private expand(g: Game): Node {
    const legal = g.legalActions();
    // Tactical shortcut: an immediate winning pawn move is always best.
    const gr = goalRow(g.turn);
    for (const a of legal) {
      if (a < CELLS && rowOf(a) === gr) {
        const node = new Node([a], new Float32Array([1]), 1);
        node.proven = 1;
        return node;
      }
    }
    const ev = this.evaluator.evaluate(g, legal);
    return new Node(legal, ev.priors, ev.value);
  }

  private ensureRoot() {
    if (!this.root) {
      this.root = this.expand(this.rootGame);
      this.root.visits = 1;
    }
  }

  private addNoise(node: Node) {
    const { dirichletAlpha: alpha, dirichletEps: eps } = this.opts;
    if (alpha <= 0 || eps <= 0 || node.actions.length < 2) return;
    const noise = node.actions.length;
    const d = new Float32Array(noise);
    let s = 0;
    for (let i = 0; i < noise; i++) {
      d[i] = sampleGamma(alpha, this.rng);
      s += d[i];
    }
    for (let i = 0; i < noise; i++) node.priors[i] = (1 - eps) * node.priors[i] + (eps * d[i]) / (s || 1);
  }

  private select(node: Node): number {
    const { cPuct, fpuReduction } = this.opts;
    const sqrtN = Math.sqrt(node.visits);
    // FPU: unvisited children are assumed a bit worse than the parent
    let visitedPrior = 0;
    for (let i = 0; i < node.N.length; i++) if (node.N[i] > 0) visitedPrior += node.priors[i];
    const parentQ = node.visits > 1 ? this.nodeQ(node) : node.value;
    const fpu = parentQ - fpuReduction * Math.sqrt(visitedPrior);
    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < node.N.length; i++) {
      const n = node.N[i];
      const q = n > 0 ? node.W[i] / n : fpu;
      const score = q + (cPuct * node.priors[i] * sqrtN) / (1 + n);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  private nodeQ(node: Node): number {
    let w = 0;
    let n = 0;
    for (let i = 0; i < node.N.length; i++) {
      w += node.W[i];
      n += node.N[i];
    }
    return n > 0 ? w / n : node.value;
  }

  /** Run `sims` simulations (adds to any existing search). */
  search(sims: number, noise = false): void {
    this.ensureRoot();
    const root = this.root!;
    if (noise) this.addNoise(root);
    if (root.proven !== null) return;
    const pathNodes: Node[] = [];
    const pathIdx: number[] = [];
    for (let s = 0; s < sims; s++) {
      const g = this.rootGame.clone();
      let node = root;
      pathNodes.length = 0;
      pathIdx.length = 0;
      let value: number;
      for (;;) {
        if (node.proven !== null) {
          value = node.proven;
          break;
        }
        const i = this.select(node);
        pathNodes.push(node);
        pathIdx.push(i);
        g.play(node.actions[i]);
        let child = node.children[i];
        if (child) {
          node = child;
          continue;
        }
        if (g.over) {
          // The player who just moved won (or draw). Value is for side to move at the new state.
          value = g.winner === 2 ? 0 : -1;
          child = new Node([], new Float32Array(0), value);
          child.proven = value;
        } else {
          child = this.expand(g);
          value = child.proven ?? child.value;
        }
        node.children[i] = child;
        node = child;
        break;
      }
      node.visits++;
      for (let k = pathNodes.length - 1; k >= 0; k--) {
        value = -value;
        const pn = pathNodes[k];
        const pi = pathIdx[k];
        pn.N[pi]++;
        pn.W[pi] += value;
        pn.visits++;
      }
    }
  }

  result(maxMoves = 10): SearchResult {
    this.ensureRoot();
    const root = this.root!;
    const idx = Array.from(root.actions.keys()).sort(
      (a, b) => root.N[b] - root.N[a] || root.priors[b] - root.priors[a],
    );
    const moves: MoveStat[] = [];
    for (const i of idx.slice(0, maxMoves)) {
      moves.push({
        action: root.actions[i],
        visits: root.N[i],
        q: root.N[i] > 0 ? root.W[i] / root.N[i] : root.proven ?? -root.value,
        prior: root.priors[i],
        pv: [root.actions[i], ...this.pv(root.children[i])],
      });
    }
    // If the root is proven, the single move carries the win
    if (root.proven !== null && moves.length) moves[0].q = root.proven;
    const value = root.proven ?? (root.visits > 1 ? this.nodeQ(root) : root.value);
    return { value, visits: root.visits, moves };
  }

  private pv(node: Node | null, depth = 8): number[] {
    const out: number[] = [];
    while (node && depth-- > 0 && node.actions.length) {
      let best = -1;
      let bn = 0;
      for (let i = 0; i < node.N.length; i++)
        if (node.N[i] > bn) {
          bn = node.N[i];
          best = i;
        }
      if (best < 0) break;
      out.push(node.actions[best]);
      node = node.children[best];
    }
    return out;
  }

  /** Visit-count distribution over root actions. */
  visitDistribution(): { actions: Int16Array; probs: Float32Array } {
    this.ensureRoot();
    const root = this.root!;
    const probs = new Float32Array(root.actions.length);
    let s = 0;
    for (let i = 0; i < probs.length; i++) s += root.N[i];
    for (let i = 0; i < probs.length; i++) probs[i] = s > 0 ? root.N[i] / s : root.priors[i];
    return { actions: root.actions, probs };
  }

  /** Choose a move: temperature 0 = most visited; otherwise sample N^(1/T). */
  pick(temperature = 0): number {
    const { actions, probs } = this.visitDistribution();
    if (temperature <= 1e-3) {
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      return actions[best];
    }
    const w = new Float64Array(probs.length);
    let s = 0;
    for (let i = 0; i < w.length; i++) {
      w[i] = Math.pow(probs[i], 1 / temperature);
      s += w[i];
    }
    let r = this.rng() * s;
    for (let i = 0; i < w.length; i++) {
      r -= w[i];
      if (r <= 0) return actions[i];
    }
    return actions[actions.length - 1];
  }

  /** Advance the root by an action, keeping the subtree (tree reuse). */
  advance(action: number): void {
    let next: Node | null = null;
    if (this.root) {
      const i = this.root.actions.indexOf(action);
      if (i >= 0) next = this.root.children[i];
    }
    this.rootGame.play(action);
    this.root = next;
    if (this.root && this.root.visits === 0) this.root.visits = 1;
  }
}

/** Marsaglia–Tsang gamma sampler (alpha < 1 handled via boost). */
export function sampleGamma(alpha: number, rng: () => number): number {
  if (alpha < 1) return sampleGamma(alpha + 1, rng) * Math.pow(rng() || 1e-12, 1 / alpha);
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      const u1 = rng() || 1e-12;
      const u2 = rng();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
