/**
 * Self-play game generation. Produces training samples:
 *   x  : encoded position (side-to-move perspective)
 *   pi : MCTS visit distribution in canonical action space
 *   z  : final outcome for the side to move (blended with root Q)
 */
import { Game, NUM_ACTIONS } from '../src/engine/quoridor.ts';
import { encode, INPUT_SIZE, toCanonical } from '../src/ai/features.ts';
import type { Evaluator } from '../src/ai/evaluator.ts';
import { MCTS } from '../src/ai/mcts.ts';

export interface Sample {
  x: Float32Array;
  /** sparse policy target: canonical actions + probabilities */
  pa: Int16Array;
  pp: Float32Array;
  /** canonical legal actions */
  legal: Int16Array;
  z: number;
}

export interface SelfPlayOptions {
  sims: number;
  /** cheaper search used on "fast" moves (playout cap randomisation, KataGo-style) */
  fastSims: number;
  /** probability a move uses the full search (only full-search moves become training targets) */
  fullProb: number;
  /** plies with temperature 1 before switching to greedy */
  tempPlies: number;
  /** weight of root Q in the value target (rest is game outcome) */
  qWeight: number;
  maxPlies: number;
  /** resign threshold on root value; disabled if <= -1 */
  resign: number;
}

export const DEFAULT_SELFPLAY: SelfPlayOptions = {
  sims: 200,
  fastSims: 40,
  fullProb: 0.35,
  tempPlies: 12,
  qWeight: 0.3,
  maxPlies: 160,
  resign: -0.97,
};

export interface GameRecord {
  samples: Sample[];
  winner: number;
  plies: number;
  moves: number[];
}

export function playGame(evaluator: Evaluator, opts: SelfPlayOptions, rng: () => number = Math.random): GameRecord {
  const g = new Game();
  g.maxPlies = opts.maxPlies;
  const pending: { x: Float32Array; pa: Int16Array; pp: Float32Array; legal: Int16Array; turn: number; q: number }[] = [];
  const moves: number[] = [];
  const mcts = new MCTS(g, evaluator, { dirichletAlpha: 0.15, dirichletEps: 0.25 }, rng);
  let resigned = -1;
  while (!g.over) {
    const full = rng() < opts.fullProb;
    // Fresh noise per move: reset the root so noise is only applied on full searches
    if (full) {
      mcts.root = null;
      mcts.search(opts.sims, true);
    } else {
      mcts.search(Math.max(1, opts.fastSims - (mcts.root?.visits ?? 0)));
    }
    const res = mcts.result(1);
    if (opts.resign > -1 && res.value < opts.resign && g.ply > 10) {
      resigned = g.turn;
      break;
    }
    if (full) {
      const { actions, probs } = mcts.visitDistribution();
      const pa = new Int16Array(actions.length);
      for (let i = 0; i < actions.length; i++) pa[i] = toCanonical(actions[i], g.turn);
      pending.push({
        x: encode(g, new Float32Array(INPUT_SIZE)),
        pa,
        pp: probs.slice(),
        legal: pa.slice(),
        turn: g.turn,
        q: res.value,
      });
    }
    const temp = g.ply < opts.tempPlies ? 1 : 0;
    const a = mcts.pick(temp);
    moves.push(a);
    mcts.advance(a);
    g.play(a);
  }
  const winner = resigned >= 0 ? 1 - resigned : g.winner;
  const samples: Sample[] = pending.map((p) => {
    const outcome = winner === 2 ? 0 : winner === p.turn ? 1 : -1;
    return { x: p.x, pa: p.pa, pp: p.pp, legal: p.legal, z: (1 - opts.qWeight) * outcome + opts.qWeight * p.q };
  });
  return { samples, winner, plies: g.ply, moves };
}

export const ACTIONS = NUM_ACTIONS;
