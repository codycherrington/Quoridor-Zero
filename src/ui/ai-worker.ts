/// <reference lib="webworker" />
import { Game } from '../engine/quoridor.ts';
import { HeuristicEvaluator, NetEvaluator, type Evaluator } from '../ai/evaluator.ts';
import { MCTS, type SearchResult } from '../ai/mcts.ts';
import { Net } from '../ai/net.ts';
import { levelById } from '../ai/levels.ts';
import type { BookEntry, WorkerRequest, WorkerResponse } from './protocol.ts';

declare const self: DedicatedWorkerGlobalScope;

let evaluator: Evaluator = new HeuristicEvaluator();
let book: Record<string, BookEntry> = {};
let modelInfo = 'heuristic (no trained model found)';
/** Bumped by an explicit 'cancel' (new game, undo…): aborts everything in flight. */
let hardGen = 0;
/** Bumped by any new request: aborts in-flight analyses (they are only ever "latest wins"). */
let softGen = 0;
type Token = { hard: number; soft: number | null };
const alive = (t: Token) => t.hard === hardGen && (t.soft === null || t.soft === softGen);

let resolveBase: (b: string) => void = () => {};
const baseUrl = new Promise<string>((r) => (resolveBase = r));

const ready = (async () => {
  // The page sends its own base URL: a worker's relative URLs resolve against the worker
  // script (…/assets/), not the page, so it can't work out where models/ lives by itself.
  const base = await baseUrl;
  try {
    const res = await fetch(new URL('models/model.json', base));
    if (res.ok) {
      const json = await res.json();
      const net = Net.fromJSON(json);
      evaluator = new NetEvaluator(net, { cacheSize: 50000 });
      const m = json.meta ?? {};
      modelInfo = `network gen ${m.generation ?? '?'} · ${Net.paramCount(net.config).toLocaleString()} params`;
    }
  } catch {
    /* fall back to heuristic */
  }
  try {
    const res = await fetch(new URL('models/book.json', base));
    if (res.ok) book = (await res.json()).positions ?? {};
  } catch {
    /* no book */
  }
  post({ type: 'ready', modelInfo, bookSize: Object.keys(book).length });
})();

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

function gameFrom(moves: number[]): Game {
  const g = new Game();
  for (const a of moves) g.play(a);
  return g;
}

const yieldToEvents = () => new Promise((r) => setTimeout(r, 0));

/** Persistent search tree for the position currently being analysed. */
let cached: { key: string; mcts: MCTS } | null = null;
function treeFor(g: Game): MCTS {
  const key = g.key() + ':' + g.ply;
  if (cached && cached.key === key) return cached.mcts;
  cached = { key, mcts: new MCTS(g, evaluator) };
  return cached.mcts;
}

async function runSearch(m: MCTS, sims: number, job: Token, onProgress?: (r: SearchResult) => void): Promise<boolean> {
  const chunk = 64;
  let lastPost = performance.now();
  while ((m.root?.visits ?? 0) < sims) {
    if (!alive(job)) return false;
    m.search(Math.min(chunk, sims - (m.root?.visits ?? 0)));
    if (m.root?.proven != null) break;
    if (onProgress && performance.now() - lastPost > 150) {
      onProgress(m.result(8));
      lastPost = performance.now();
    }
    await yieldToEvents();
  }
  return true;
}

function bookResult(g: Game): SearchResult | null {
  const e = book[g.key()];
  if (!e) return null;
  return { value: e.value, visits: e.visits, moves: e.moves.map((m) => ({ ...m, prior: 0 })) };
}

// Requests run one at a time, in order. Analyses are abandoned as soon as anything newer arrives.
let chain: Promise<void> = ready;
self.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  if (req.type === 'init') {
    resolveBase(req.base);
    return;
  }
  softGen++;
  if (req.type === 'cancel') {
    hardGen++;
    return;
  }
  const job: Token = { hard: hardGen, soft: req.type === 'analyze' ? softGen : null };
  chain = chain.then(() => handle(req, job)).catch((e) => console.error(e));
};

async function handle(req: Exclude<WorkerRequest, { type: 'cancel' } | { type: 'init' }>, job: Token): Promise<void> {
  if (!alive(job)) return;
  const g = gameFrom(req.moves);
  if (g.over) return;

  if (req.type === 'move') {
    const level = levelById(req.level);
    const m = new MCTS(g, evaluator);
    const started = performance.now();
    if (!(await runSearch(m, level.sims, job))) return;
    // Don't answer instantly — a tiny delay feels more natural
    const elapsed = performance.now() - started;
    if (elapsed < 350) await new Promise((r) => setTimeout(r, 350 - elapsed));
    if (!alive(job)) return;
    const temp = g.ply < level.tempPlies ? level.temperature : 0;
    const action = m.pick(temp);
    post({ type: 'move', id: req.id, action, result: m.result(5) });
    return;
  }

  if (req.type === 'analyze') {
    const fromBook = req.useBook !== false ? bookResult(g) : null;
    if (fromBook) {
      post({ type: 'analysis', id: req.id, result: fromBook, done: true, source: 'book' });
      return;
    }
    const m = treeFor(g);
    const ok = await runSearch(m, req.sims, job, (r) => post({ type: 'analysis', id: req.id, result: r, done: false, source: 'search' }));
    if (!ok) return;
    post({ type: 'analysis', id: req.id, result: m.result(8), done: true, source: 'search' });
    return;
  }

  if (req.type === 'grade') {
    const m = treeFor(g);
    if (!(await runSearch(m, req.sims, job))) return;
    const res = m.result(400);
    const best = res.moves[0];
    let played = res.moves.find((x) => x.action === req.action);
    let playedQ: number;
    if (played && played.visits >= Math.max(12, req.sims / 40)) {
      playedQ = played.q;
    } else {
      // The search barely looked at the played move: evaluate the resulting position directly.
      const child = g.clone();
      child.play(req.action);
      if (child.over) playedQ = child.winner === g.turn ? 1 : child.winner === 2 ? 0 : -1;
      else {
        const cm = new MCTS(child, evaluator);
        if (!(await runSearch(cm, Math.max(200, req.sims >> 2), job))) return;
        playedQ = -cm.result(1).value;
      }
    }
    if (!played) played = { action: req.action, visits: 0, q: playedQ, prior: 0, pv: [req.action] };
    post({ type: 'grade', id: req.id, action: req.action, bestAction: best.action, bestQ: best.q, playedQ, bestPv: best.pv, result: m.result(5) });
  }
}
