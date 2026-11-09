/**
 * Builds an opening book by running a deep search on every position reachable through the
 * engine's preferred moves for the first few plies.
 *   node training/build-book.ts --model public/models/model.json --sims 4000 --depth 9
 */
import fs from 'node:fs';
import { parseArgs } from 'node:util';
import { Game } from '../src/engine/quoridor.ts';
import { actionToString } from '../src/engine/notation.ts';
import type { SearchResult } from '../src/ai/mcts.ts';
import { runJobs } from './pool.ts';
import type { Job } from './worker.ts';

const { values: args } = parseArgs({
  options: {
    model: { type: 'string', default: 'public/models/model.json' },
    sims: { type: 'string', default: '4000' },
    depth: { type: 'string', default: '9' },
    out: { type: 'string', default: 'public/models/book.json' },
    threads: { type: 'string' },
  },
});
const model = fs.readFileSync(args.model!, 'utf8');
const sims = Number(args.sims);
const depth = Number(args.depth);
/** how many candidate moves to branch on at each ply */
const branching = (ply: number) => (ply < 2 ? 3 : ply < 6 ? 2 : 1);
const MIN_SHARE = 0.06;

const positions: Record<string, { value: number; visits: number; moves: { action: number; visits: number; q: number; pv: number[] }[] }> = {};
let frontier: number[][] = [[]];
const t0 = Date.now();
for (let ply = 0; ply < depth && frontier.length; ply++) {
  const results: { moves: number[]; key: string; result: SearchResult }[] = [];
  const jobs: Job[] = [];
  const per = 4;
  for (let i = 0; i < frontier.length; i += per)
    jobs.push({ kind: 'analyze', model, positions: frontier.slice(i, i + per), sims, seed: (Math.random() * 2 ** 31) | 0 });
  await runJobs(jobs, (m) => m.type === 'analysis' && results.push(m), args.threads ? Number(args.threads) : undefined);
  const next: number[][] = [];
  const seen = new Set<string>();
  for (const { moves, key, result } of results) {
    positions[key] = {
      value: +result.value.toFixed(4),
      visits: result.visits,
      moves: result.moves
        .filter((m) => m.visits > 0)
        .map((m) => ({ action: m.action, visits: m.visits, q: +m.q.toFixed(4), pv: m.pv.slice(0, 8) })),
    };
    const total = Math.max(1, result.visits - 1);
    const cands = result.moves.filter((m) => m.visits / total >= MIN_SHARE).slice(0, branching(ply));
    for (const c of cands) {
      const line = [...moves, c.action];
      const g = new Game();
      for (const a of line) g.play(a);
      const k = g.key();
      if (g.over || positions[k] || seen.has(k)) continue;
      seen.add(k);
      next.push(line);
    }
  }
  console.log(`ply ${ply}: analysed ${frontier.length} positions, ${Object.keys(positions).length} total (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  frontier = next;
}
const root = positions[new Game().key()];
console.log('root:', root.moves.map((m) => `${actionToString(m.action)} ${m.visits}`).join(', '));
fs.writeFileSync(args.out!, JSON.stringify({ format: 'quoridor-zero-book-v1', sims, depth, created: new Date().toISOString(), positions }));
console.log(`wrote ${args.out} (${Object.keys(positions).length} positions)`);
