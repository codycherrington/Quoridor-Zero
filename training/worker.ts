import { parentPort } from 'node:worker_threads';
import { HeuristicEvaluator, NetEvaluator, type Evaluator } from '../src/ai/evaluator.ts';
import { Net } from '../src/ai/net.ts';
import { playGame, type SelfPlayOptions } from './selfplay.ts';
import { playMatch, type AgentSpec } from './match.ts';

export type Job =
  | { kind: 'selfplay'; model: string | null; heuristicBlend: number; games: number; opts: SelfPlayOptions; seed: number }
  | { kind: 'match'; a: AgentSpec; b: AgentSpec; games: number; firstAsA: number; seed: number };

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

parentPort!.on('message', (job: Job) => {
  const rng = mulberry32(job.seed);
  if (job.kind === 'selfplay') {
    const ev: Evaluator = job.model
      ? new NetEvaluator(Net.fromJSON(job.model), { heuristicBlend: job.heuristicBlend })
      : new HeuristicEvaluator();
    for (let i = 0; i < job.games; i++) {
      const rec = playGame(ev, job.opts, rng);
      parentPort!.postMessage({ type: 'game', rec });
    }
    parentPort!.postMessage({ type: 'done' });
  } else {
    for (let i = 0; i < job.games; i++) {
      const aFirst = (job.firstAsA + i) % 2 === 0;
      const res = playMatch(job.a, job.b, aFirst, rng);
      parentPort!.postMessage({ type: 'match', aFirst, ...res });
    }
    parentPort!.postMessage({ type: 'done' });
  }
});
