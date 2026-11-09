import { CELLS, Game, H_OFFSET, V_OFFSET, wallTouches } from '../engine/quoridor.ts';
import { encode, INPUT_SIZE, toCanonical } from './features.ts';
import type { Net } from './net.ts';

export interface Evaluation {
  /** prior probabilities aligned with the `legal` array passed in */
  priors: Float32Array;
  /** value in [-1, 1] for the side to move */
  value: number;
}

export interface Evaluator {
  evaluate(g: Game, legal: number[]): Evaluation;
}

/** Quick hand-crafted value: path-length race + wall reserve, from side-to-move's view. */
export function heuristicValue(g: Game): number {
  const me = g.turn;
  const dm = g.distance(me);
  const dop = g.distance(1 - me);
  const wallDiff = g.walls[me] - g.walls[1 - me];
  // +0.5 tempo for being on move
  return Math.tanh(0.32 * (dop - dm + 0.5) + 0.08 * wallDiff);
}

/**
 * Hand-crafted evaluator. Used to bootstrap training (so self-play does not start from random noise)
 * and as the engine behind the "classic" AI.
 */
export class HeuristicEvaluator implements Evaluator {
  evaluate(g: Game, legal: number[]): Evaluation {
    const me = g.turn;
    const opp = 1 - me;
    const priors = new Float32Array(legal.length);
    const myField = g.distanceField(me);
    const myDist = myField[g.pos[me]];
    const oppDist = g.distance(opp);
    const oppPath = g.pathEdges(opp);
    // Copy the edges: pathEdges returns shared buffers
    const oppEdges = { up: oppPath.up.slice(), right: oppPath.right.slice() };

    for (let i = 0; i < legal.length; i++) {
      const a = legal[i];
      if (a < CELLS) {
        const d = myField[a];
        priors[i] = d < myDist ? 5 : d === myDist ? 0.8 : 0.25;
        continue;
      }
      const horizontal = a < V_OFFSET;
      const w = horizontal ? a - H_OFFSET : a - V_OFFSET;
      if (!wallTouches(horizontal, w, oppEdges)) {
        priors[i] = 0.02;
        continue;
      }
      g.setWall(horizontal, w, 1);
      const gain = g.distance(opp) - oppDist - (g.distance(me) - myDist);
      g.setWall(horizontal, w, 0);
      priors[i] = gain > 0 ? 0.6 * Math.exp(0.7 * Math.min(gain, 6)) : gain === 0 ? 0.15 : 0.03;
    }
    normalize(priors);
    return { priors, value: heuristicValue(g) };
  }
}

/** Neural-network evaluator (AlphaZero style). Optionally blends in the heuristic value. */
export class NetEvaluator implements Evaluator {
  private x = new Float32Array(INPUT_SIZE);
  private cache = new Map<string, { logits: Float32Array; value: number }>();
  net: Net;
  /** 0 = pure network value, 1 = pure heuristic value */
  heuristicBlend: number;
  /** softmax temperature on the policy logits */
  policyTemp: number;
  cacheSize: number;

  constructor(net: Net, opts: { heuristicBlend?: number; policyTemp?: number; cacheSize?: number } = {}) {
    this.net = net;
    this.heuristicBlend = opts.heuristicBlend ?? 0;
    this.policyTemp = opts.policyTemp ?? 1;
    this.cacheSize = opts.cacheSize ?? 0;
  }

  raw(g: Game): { logits: Float32Array; value: number } {
    let key = '';
    if (this.cacheSize > 0) {
      key = g.key();
      const hit = this.cache.get(key);
      if (hit) return hit;
    }
    encode(g, this.x);
    const out = this.net.predict(this.x);
    const res = { logits: out.logits.slice(), value: out.value };
    if (this.cacheSize > 0) {
      if (this.cache.size >= this.cacheSize) this.cache.clear();
      this.cache.set(key, res);
    }
    return res;
  }

  evaluate(g: Game, legal: number[]): Evaluation {
    const { logits, value } = this.raw(g);
    const priors = new Float32Array(legal.length);
    let mx = -Infinity;
    for (let i = 0; i < legal.length; i++) {
      const l = logits[toCanonical(legal[i], g.turn)];
      priors[i] = l;
      if (l > mx) mx = l;
    }
    const invT = 1 / this.policyTemp;
    for (let i = 0; i < legal.length; i++) priors[i] = Math.exp((priors[i] - mx) * invT);
    normalize(priors);
    let v = value;
    if (this.heuristicBlend > 0) v = (1 - this.heuristicBlend) * v + this.heuristicBlend * heuristicValue(g);
    return { priors, value: v };
  }
}

export function normalize(p: Float32Array) {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += p[i];
  if (s <= 0) {
    p.fill(1 / p.length);
    return;
  }
  for (let i = 0; i < p.length; i++) p[i] /= s;
}
