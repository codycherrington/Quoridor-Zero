/**
 * AlphaZero-style training loop:
 *   repeat:
 *     1. self-play games with the current network + MCTS (parallel worker threads)
 *     2. add positions to a sliding replay window
 *     3. train the network on random mini-batches (with left/right mirror augmentation)
 *     4. gate: play the new net against the previous best; keep it only if it scores well
 *
 * Bootstrap: the first `--bootstrap` iterations use a hand-crafted evaluator (path-race heuristic)
 * instead of the (still random) network, so self-play starts from sensible games.
 *
 * Usage: node training/train.ts --iters 20 --games 64 --sims 200 [--resume path] [--out public/models/model.json]
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { INPUT_SIZE, MIRROR_PERM, mirrorInput } from '../src/ai/features.ts';
import { DEFAULT_CONFIG, Net, type TrainBatch } from '../src/ai/net.ts';
import { NUM_ACTIONS } from '../src/engine/quoridor.ts';
import { runJobs } from './pool.ts';
import { DEFAULT_SELFPLAY, type GameRecord, type Sample } from './selfplay.ts';
import type { Job } from './worker.ts';

const { values: args } = parseArgs({
  options: {
    iters: { type: 'string', default: '20' },
    games: { type: 'string', default: '64' },
    sims: { type: 'string', default: '200' },
    bootstrap: { type: 'string', default: '2' },
    'bootstrap-games': { type: 'string', default: '200' },
    steps: { type: 'string', default: '400' },
    batch: { type: 'string', default: '256' },
    lr: { type: 'string', default: '0.001' },
    window: { type: 'string', default: '120000' },
    blend: { type: 'string', default: '0.5' },
    'gate-games': { type: 'string', default: '40' },
    'gate-sims': { type: 'string', default: '100' },
    threads: { type: 'string' },
    until: { type: 'string' },
    resume: { type: 'string' },
    out: { type: 'string', default: 'public/models/model.json' },
    runs: { type: 'string', default: 'training/runs' },
  },
});

const iters = Number(args.iters);
const gamesPerIter = Number(args.games);
const sims = Number(args.sims);
const bootstrapIters = Number(args.bootstrap);
const steps = Number(args.steps);
const batchSize = Number(args.batch);
const baseLr = Number(args.lr);
const windowSize = Number(args.window);
const blend0 = Number(args.blend);
const gateGames = Number(args['gate-games']);
const gateSims = Number(args['gate-sims']);
const threads = args.threads ? Number(args.threads) : undefined;
fs.mkdirSync(args.runs!, { recursive: true });

const resumeJson = args.resume ? fs.readFileSync(args.resume, 'utf8') : null;
let net = resumeJson ? Net.fromJSON(resumeJson) : new Net(DEFAULT_CONFIG);
let generation = resumeJson ? Number(JSON.parse(resumeJson).meta?.generation ?? 0) : 0;
let best = resumeJson ?? net.toJSON();
const buffer: Sample[] = [];
const log = (s: string) => {
  const line = `[${new Date().toISOString().slice(11, 19)}] ${s}`;
  console.log(line);
  fs.appendFileSync(path.join(args.runs!, 'train.log'), line + '\n');
};

async function selfPlay(model: string | null, games: number, heuristicBlend: number): Promise<GameRecord[]> {
  const recs: GameRecord[] = [];
  const per = 4;
  const jobs: Job[] = [];
  for (let i = 0; i < games; i += per)
    jobs.push({
      kind: 'selfplay',
      model,
      heuristicBlend,
      games: Math.min(per, games - i),
      opts: { ...DEFAULT_SELFPLAY, sims, fastSims: Math.max(16, sims >> 2) },
      seed: (Math.random() * 2 ** 31) | 0,
    });
  await runJobs(jobs, (m) => m.type === 'game' && recs.push(m.rec), threads);
  return recs;
}

async function gate(candidate: string, incumbent: string): Promise<number> {
  let score = 0;
  let n = 0;
  const jobs: Job[] = [];
  const per = 4;
  for (let i = 0; i < gateGames; i += per)
    jobs.push({
      kind: 'match',
      a: { model: candidate, sims: gateSims },
      b: { model: incumbent, sims: gateSims },
      games: Math.min(per, gateGames - i),
      firstAsA: i,
      seed: (Math.random() * 2 ** 31) | 0,
    });
  await runJobs(
    jobs,
    (m) => {
      if (m.type === 'match') {
        score += m.score;
        n++;
      }
    },
    threads,
  );
  return n ? score / n : 0.5;
}

function makeBatch(B: number): TrainBatch {
  const x = new Float32Array(B * INPUT_SIZE);
  const pi = new Float32Array(B * NUM_ACTIONS);
  const mask = new Uint8Array(B * NUM_ACTIONS);
  const z = new Float32Array(B);
  const tmp = new Float32Array(INPUT_SIZE);
  for (let s = 0; s < B; s++) {
    const smp = buffer[(Math.random() * buffer.length) | 0];
    const mirror = Math.random() < 0.5;
    const o = s * NUM_ACTIONS;
    if (mirror) {
      mirrorInput(smp.x, tmp);
      x.set(tmp, s * INPUT_SIZE);
    } else x.set(smp.x, s * INPUT_SIZE);
    for (let i = 0; i < smp.legal.length; i++) {
      const a = mirror ? MIRROR_PERM[smp.legal[i]] : smp.legal[i];
      mask[o + a] = 1;
    }
    for (let i = 0; i < smp.pa.length; i++) {
      const a = mirror ? MIRROR_PERM[smp.pa[i]] : smp.pa[i];
      pi[o + a] = smp.pp[i];
    }
    z[s] = smp.z;
  }
  return { x, pi, mask, z, B };
}

function summarize(recs: GameRecord[]) {
  const w = [0, 0, 0];
  let plies = 0;
  let samples = 0;
  for (const r of recs) {
    w[r.winner === 2 ? 2 : r.winner]++;
    plies += r.plies;
    samples += r.samples.length;
  }
  return `games=${recs.length} p1=${w[0]} p2=${w[1]} draws=${w[2]} avgPlies=${(plies / recs.length).toFixed(1)} samples=${samples}`;
}

const stopAt = args.until ? Date.parse(args.until) : Infinity;
for (let it = 0; it < iters; it++) {
  if (Date.now() >= stopAt) {
    log(`reached --until ${args.until}; stopping at generation ${generation}`);
    break;
  }
  const t0 = Date.now();
  const useHeuristic = it < bootstrapIters && !args.resume;
  // Heuristic blending in the value decays as training progresses
  const blend = useHeuristic ? 0 : Math.max(0, blend0 * (1 - it / Math.max(1, iters - 1)));
  const recs = await selfPlay(useHeuristic ? null : best, useHeuristic ? Number(args['bootstrap-games']) : gamesPerIter, blend);
  for (const r of recs) buffer.push(...r.samples);
  if (buffer.length > windowSize) buffer.splice(0, buffer.length - windowSize);
  log(`iter ${it} ${useHeuristic ? '(bootstrap) ' : `(blend ${blend.toFixed(2)}) `}${summarize(recs)} buffer=${buffer.length} selfplay=${((Date.now() - t0) / 1000).toFixed(0)}s`);

  const t1 = Date.now();
  const nSteps = Math.min(steps, Math.ceil((buffer.length * 2) / batchSize));
  let pl = 0;
  let vl = 0;
  for (let s = 0; s < nSteps; s++) {
    const lr = baseLr * (s < 20 && it === 0 ? (s + 1) / 20 : 1);
    const r = net.trainStep(makeBatch(batchSize), lr);
    pl += r.policyLoss;
    vl += r.valueLoss;
  }
  log(`  train steps=${nSteps} policyLoss=${(pl / nSteps).toFixed(3)} valueLoss=${(vl / nSteps).toFixed(3)} (${((Date.now() - t1) / 1000).toFixed(0)}s)`);

  const candidate = net.toJSON({ generation: generation + 1, iter: it, samples: buffer.length, date: new Date().toISOString() });
  let accept = true;
  if (!useHeuristic && gateGames > 0 && generation > 0) {
    const sc = await gate(candidate, best);
    accept = sc >= 0.5;
    log(`  gate vs gen ${generation}: score=${(sc * 100).toFixed(1)}% -> ${accept ? 'ACCEPT' : 'reject'}`);
  }
  if (accept) {
    generation++;
    best = net.toJSON({ generation, iter: it, samples: buffer.length, date: new Date().toISOString() });
    fs.writeFileSync(path.join(args.runs!, `gen${String(generation).padStart(3, '0')}.json`), best);
    fs.mkdirSync(path.dirname(args.out!), { recursive: true });
    fs.writeFileSync(args.out!, best);
    log(`  saved generation ${generation} -> ${args.out}`);
  } else {
    // keep training the candidate weights (more data next iteration) but self-play with the incumbent
  }
}
