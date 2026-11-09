import { Game } from '../src/engine/quoridor.ts';
import { HeuristicEvaluator, NetEvaluator, type Evaluator } from '../src/ai/evaluator.ts';
import { MCTS } from '../src/ai/mcts.ts';
import { Net } from '../src/ai/net.ts';

export interface AgentSpec {
  /** model JSON string, or null for the heuristic evaluator */
  model: string | null;
  sims: number;
  heuristicBlend?: number;
}

const cache = new Map<string, Evaluator>();
function evaluatorFor(spec: AgentSpec): Evaluator {
  const key = (spec.model ? spec.model.slice(-64) + spec.model.length : 'heuristic') + ':' + (spec.heuristicBlend ?? 0);
  let ev = cache.get(key);
  if (!ev) {
    ev = spec.model
      ? new NetEvaluator(Net.fromJSON(spec.model), { heuristicBlend: spec.heuristicBlend ?? 0 })
      : new HeuristicEvaluator();
    cache.set(key, ev);
  }
  return ev;
}

/** Plays one game. Returns winner relative to A: 1 = A won, 0 = B won, 0.5 = draw. */
export function playMatch(a: AgentSpec, b: AgentSpec, aFirst: boolean, rng: () => number): { score: number; plies: number } {
  const g = new Game();
  g.maxPlies = 200;
  const agents = aFirst ? [a, b] : [b, a];
  while (!g.over) {
    const spec = agents[g.turn];
    const m = new MCTS(g, evaluatorFor(spec), {}, rng);
    m.search(spec.sims);
    // a little opening variety so matches are not all identical
    const act = m.pick(g.ply < 4 ? 0.5 : 0);
    g.play(act);
  }
  if (g.winner === 2) return { score: 0.5, plies: g.ply };
  const aSeat = aFirst ? 0 : 1;
  return { score: g.winner === aSeat ? 1 : 0, plies: g.ply };
}
