/**
 * Head-to-head match between two agents.
 *   node training/arena.ts --a public/models/model.json --b heuristic --games 40 --sims 200
 */
import fs from 'node:fs';
import { parseArgs } from 'node:util';
import { runJobs } from './pool.ts';
import type { Job } from './worker.ts';

const { values: args } = parseArgs({
  options: {
    a: { type: 'string', default: 'public/models/model.json' },
    b: { type: 'string', default: 'heuristic' },
    games: { type: 'string', default: '40' },
    sims: { type: 'string', default: '200' },
    'sims-b': { type: 'string' },
    'blend-a': { type: 'string', default: '0' },
    threads: { type: 'string' },
  },
});
const load = (s: string) => (s === 'heuristic' ? null : fs.readFileSync(s, 'utf8'));
const games = Number(args.games);
const jobs: Job[] = [];
for (let i = 0; i < games; i += 2)
  jobs.push({
    kind: 'match',
    a: { model: load(args.a!), sims: Number(args.sims), heuristicBlend: Number(args['blend-a']) },
    b: { model: load(args.b!), sims: Number(args['sims-b'] ?? args.sims) },
    games: Math.min(2, games - i),
    firstAsA: 0,
    seed: (Math.random() * 2 ** 31) | 0,
  });
let score = 0;
let n = 0;
const asFirst = [0, 0];
const asSecond = [0, 0];
const t = Date.now();
await runJobs(
  jobs,
  (m) => {
    if (m.type !== 'match') return;
    score += m.score;
    n++;
    if (m.aFirst) (asFirst[0] += m.score), asFirst[1]++;
    else (asSecond[0] += m.score), asSecond[1]++;
  },
  args.threads ? Number(args.threads) : undefined,
);
const elo = (s: number) => (s <= 0 ? -Infinity : s >= 1 ? Infinity : -400 * Math.log10(1 / s - 1));
console.log(`A=${args.a} vs B=${args.b}: A scored ${score}/${n} (${((100 * score) / n).toFixed(1)}%, Elo ${elo(score / n).toFixed(0)})`);
console.log(`  A as first player: ${asFirst[0]}/${asFirst[1]}   A as second player: ${asSecond[0]}/${asSecond[1]}   (${((Date.now() - t) / 1000).toFixed(0)}s)`);
