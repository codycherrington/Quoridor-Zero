import './style.css';
import { CELLS, Game } from '../engine/quoridor.ts';
import { actionToString } from '../engine/notation.ts';
import { LEVELS, levelById } from '../ai/levels.ts';
import type { MoveStat, SearchResult } from '../ai/mcts.ts';
import { BoardUI, type Marker } from './board.ts';
import type { WorkerRequest, WorkerResponse } from './protocol.ts';

type Mode = 'ai' | 'local' | 'coach' | 'explore';
type Grade = 'best' | 'excellent' | 'good' | 'inaccuracy' | 'mistake' | 'blunder';

interface Feedback {
  ply: number;
  action: number;
  grade: Grade;
  drop: number; // win-probability points lost (0..1)
  bestAction: number;
  bestPv: number[];
  playedWin: number;
  bestWin: number;
}

const GRADE_INFO: Record<Grade, { label: string; symbol: string }> = {
  best: { label: 'Best move', symbol: '★' },
  excellent: { label: 'Excellent', symbol: '!' },
  good: { label: 'Good', symbol: '' },
  inaccuracy: { label: 'Inaccuracy', symbol: '?!' },
  mistake: { label: 'Mistake', symbol: '?' },
  blunder: { label: 'Blunder', symbol: '??' },
};

const OPENING_PLIES = 12;
const ANALYSIS_SIMS = 2500;
const HINT_SIMS = 1500;
const GRADE_SIMS = 1500;

// ---------------- state ----------------
const saved = loadSettings();
const state = {
  mode: (saved.mode as Mode) ?? 'ai',
  humanSide: (saved.humanSide as number) ?? 0,
  sidePref: (saved.sidePref as 'white' | 'black' | 'random') ?? 'white',
  level: (saved.level as number) ?? 3,
  moves: [] as number[],
  redo: [] as number[],
  flipped: false,
  liveAnalysis: (saved.liveAnalysis as boolean) ?? false,
  hint: null as SearchResult | null,
  analysis: null as { result: SearchResult; source: 'book' | 'search'; done: boolean; ply: number } | null,
  feedback: new Map<number, Feedback>(),
  thinking: false,
  grading: false,
  modelInfo: 'loading…',
  /** Openings mode: free exploration or a drill where you must find good moves for one side */
  drillOn: (saved.drillOn as boolean) ?? false,
  drillSide: (saved.drillSide as number) ?? 0,
  drill: { right: 0, total: 0, attempted: false, msg: '', cls: '' },
};
const DRILL_PLIES = 14;
const DRILL_TOLERANCE = 0.04; // win-probability points a move may lose and still count as correct

function loadSettings(): Record<string, unknown> {
  try {
    return JSON.parse(localStorage.getItem('qz-settings') ?? '{}');
  } catch {
    return {};
  }
}
function saveSettings() {
  try {
    const { mode, humanSide, sidePref, level, liveAnalysis, drillOn, drillSide } = state;
    localStorage.setItem('qz-settings', JSON.stringify({ mode, humanSide, sidePref, level, liveAnalysis, drillOn, drillSide }));
  } catch {
    /* storage unavailable */
  }
}

// ---------------- worker ----------------
const worker = new Worker(new URL('./ai-worker.ts', import.meta.url), { type: 'module' });
worker.postMessage({ type: 'init', base: new URL(import.meta.env.BASE_URL, location.href).href } satisfies WorkerRequest);
let reqId = 0;
const pending = new Map<number, (msg: WorkerResponse) => void>();
type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
function send(req: DistOmit<Exclude<WorkerRequest, { type: 'cancel' } | { type: 'init' }>, 'id'>, handler: (msg: WorkerResponse) => void): number {
  const id = ++reqId;
  pending.set(id, handler);
  worker.postMessage({ ...req, id });
  return id;
}
function cancelAll() {
  pending.clear();
  worker.postMessage({ type: 'cancel' } satisfies WorkerRequest);
  state.thinking = false;
  state.grading = false;
}
worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
  const msg = ev.data;
  if (msg.type === 'ready') {
    state.modelInfo = `Engine: ${msg.modelInfo}${msg.bookSize ? ` · opening book: ${msg.bookSize} positions` : ''}`;
    $('model-info').textContent = state.modelInfo;
    return;
  }
  const h = pending.get(msg.id);
  if (!h) return;
  if (msg.type !== 'analysis' || msg.done) pending.delete(msg.id);
  h(msg);
};

// ---------------- helpers ----------------
const $ = (id: string) => document.getElementById(id)!;
const board = new BoardUI($('board'));
board.onAction = (a) => humanMove(a);

function currentGame(): Game {
  const g = new Game();
  for (const a of state.moves) g.play(a);
  return g;
}
const winPct = (q: number) => Math.round(((q + 1) / 2) * 100);
const drilling = () => state.mode === 'explore' && state.drillOn;
const drillDone = () => drilling() && state.moves.length >= DRILL_PLIES;
const isAiTurn = (g: Game) =>
  !g.over &&
  (((state.mode === 'ai' || state.mode === 'coach') && g.turn !== state.humanSide) ||
    (drilling() && g.turn !== state.drillSide && !drillDone()));
const playerName = (p: number) => (p === 0 ? 'White' : 'Black');
const fmtPv = (pv: number[]) => pv.slice(0, 6).map(actionToString).join(' ');

function gradeFor(drop: number, isBest: boolean): Grade {
  if (isBest) return 'best';
  if (drop <= 0.02) return 'excellent';
  if (drop <= 0.05) return 'good';
  if (drop <= 0.1) return 'inaccuracy';
  if (drop <= 0.2) return 'mistake';
  return 'blunder';
}

// ---------------- game flow ----------------
function humanMove(a: number) {
  const g = currentGame();
  if (g.over || isAiTurn(g) || state.thinking || state.grading || drillDone() || !g.isLegal(a)) return;
  const before = state.moves.slice();
  if (drilling()) return drillMove(before, a);
  state.moves.push(a);
  state.redo = [];
  state.hint = null;
  state.analysis = null;
  if (state.mode === 'coach') requestGrade(before, a);
  refresh();
}

function drillMove(before: number[], action: number) {
  state.moves.push(action);
  state.grading = true;
  state.hint = null;
  state.drill.msg = '';
  renderBoard();
  renderPanels();
  send({ type: 'grade', moves: before, action, sims: GRADE_SIMS }, (msg) => {
    if (msg.type !== 'grade') return;
    state.grading = false;
    const drop = Math.max(0, (msg.bestQ - msg.playedQ) / 2);
    const ok = msg.bestAction === action || drop <= DRILL_TOLERANCE;
    const d = state.drill;
    if (!d.attempted) {
      d.total++;
      if (ok) d.right++;
    }
    const name = actionToString(action);
    if (ok) {
      d.attempted = false;
      d.cls = 'g-best';
      d.msg =
        msg.bestAction === action
          ? `✓ <b>${name}</b> — the engine's top choice.`
          : `✓ <b>${name}</b> works (engine slightly prefers <b>${actionToString(msg.bestAction)}</b>).`;
      refresh();
    } else {
      d.attempted = true;
      d.cls = 'g-mistake';
      d.msg = `✗ <b>${name}</b> gives away about ${Math.round(drop * 100)}% win chance. Try again — or ask for a hint.`;
      state.moves.pop();
      renderBoard();
      renderPanels();
    }
  });
}

/** Drill opponent: plays a varied but sound reply, preferring the opening book. */
function drillReply() {
  state.thinking = true;
  const expectedLen = state.moves.length;
  send({ type: 'analyze', moves: state.moves.slice(), sims: 1200 }, (msg) => {
    if (msg.type !== 'analysis' || !msg.done) return;
    const moves = msg.result.moves.filter((m) => m.visits > 0);
    const best = moves[0];
    const cands = moves.filter((m) => m.q >= best.q - 0.06);
    const weights = cands.map((m) => m.visits * m.visits);
    let r = Math.random() * weights.reduce((x, y) => x + y, 0);
    let pick = cands[0];
    for (let i = 0; i < cands.length; i++) if ((r -= weights[i]) <= 0) {
      pick = cands[i];
      break;
    }
    setTimeout(() => {
      state.thinking = false;
      if (state.moves.length !== expectedLen || !drilling()) return;
      state.moves.push(pick.action);
      refresh();
    }, 350);
  });
  renderPanels();
}

function requestGrade(before: number[], action: number) {
  const ply = before.length;
  state.grading = true;
  send({ type: 'grade', moves: before, action, sims: GRADE_SIMS }, (msg) => {
    if (msg.type !== 'grade') return;
    state.grading = false;
    const bestWin = (msg.bestQ + 1) / 2;
    const playedWin = (msg.playedQ + 1) / 2;
    const drop = Math.max(0, bestWin - playedWin);
    state.feedback.set(ply, {
      ply,
      action,
      grade: gradeFor(drop, msg.bestAction === action),
      drop,
      bestAction: msg.bestAction,
      bestPv: msg.bestPv,
      playedWin,
      bestWin,
    });
    renderPanels();
  });
}

function maybeAct() {
  const g = currentGame();
  if (g.over) return;
  if (isAiTurn(g) && drilling()) {
    if (!state.thinking) drillReply();
    return;
  }
  if (drilling()) return;
  if (isAiTurn(g)) {
    if (state.thinking) return;
    state.thinking = true;
    const expectedLen = state.moves.length;
    send({ type: 'move', moves: state.moves.slice(), level: state.level }, (msg) => {
      if (msg.type !== 'move') return;
      state.thinking = false;
      if (state.moves.length !== expectedLen) return;
      state.moves.push(msg.action);
      refresh();
    });
    renderPanels();
    return;
  }
  const wantAnalysis = state.mode === 'explore' || state.liveAnalysis;  // (drilling returned above)
  if (wantAnalysis) {
    const ply = state.moves.length;
    send({ type: 'analyze', moves: state.moves.slice(), sims: ANALYSIS_SIMS }, (msg) => {
      if (msg.type !== 'analysis' || state.moves.length !== ply) return;
      state.analysis = { result: msg.result, source: msg.source, done: msg.done, ply };
      renderBoard();
      renderAnalysis();
      renderEval();
    });
  } else if (state.mode === 'coach') {
    // warm the search tree while the human thinks, so grading is quick and accurate
    send({ type: 'analyze', moves: state.moves.slice(), sims: GRADE_SIMS, useBook: false }, () => {});
  }
}

function newGame() {
  cancelAll();
  state.moves = [];
  state.redo = [];
  state.hint = null;
  state.analysis = null;
  state.feedback.clear();
  state.drill = { right: 0, total: 0, attempted: false, msg: '', cls: '' };
  if (drilling()) state.flipped = state.drillSide === 1;
  else if (state.mode === 'ai' || state.mode === 'coach') {
    state.humanSide = state.sidePref === 'random' ? (Math.random() < 0.5 ? 0 : 1) : state.sidePref === 'black' ? 1 : 0;
    state.flipped = state.humanSide === 1;
  } else state.flipped = false;
  saveSettings();
  refresh();
}

function undo() {
  if (!state.moves.length) return;
  cancelAll();
  const vsAi = state.mode === 'ai' || state.mode === 'coach' || drilling();
  const side = drilling() ? state.drillSide : state.humanSide;
  do {
    const a = state.moves.pop()!;
    state.redo.push(a);
    state.feedback.delete(state.moves.length);
  } while (vsAi && state.moves.length && currentGame().turn !== side);
  state.drill.msg = '';
  state.drill.attempted = false;
  state.hint = null;
  state.analysis = null;
  refresh();
}

function redo() {
  if (!state.redo.length) return;
  cancelAll();
  if (drilling()) return;
  const vsAi = state.mode === 'ai' || state.mode === 'coach';
  do {
    state.moves.push(state.redo.pop()!);
  } while (vsAi && state.redo.length && currentGame().turn !== state.humanSide);
  state.hint = null;
  state.analysis = null;
  refresh();
}

function hint() {
  const g = currentGame();
  if (g.over || isAiTurn(g)) return;
  const ply = state.moves.length;
  $('btn-hint').textContent = 'Thinking…';
  send({ type: 'analyze', moves: state.moves.slice(), sims: HINT_SIMS }, (msg) => {
    if (msg.type !== 'analysis' || state.moves.length !== ply) return;
    if (msg.done) $('btn-hint').textContent = 'Hint';
    state.hint = msg.result;
    state.analysis = { result: msg.result, source: msg.source, done: msg.done, ply };
    renderBoard();
    renderAnalysis();
    renderEval();
  });
}

// ---------------- rendering ----------------
function refresh() {
  renderBoard();
  renderPanels();
  maybeAct();
}

function markersFrom(result: SearchResult | null, n: number): Marker[] {
  if (!result) return [];
  return result.moves
    .filter((m) => m.visits > 0 || result.moves.length === 1)
    .slice(0, n)
    .map((m, i) => ({ action: m.action, label: String(i + 1), rank: i }));
}

function renderBoard() {
  const g = currentGame();
  const owners = new Map<number, number>();
  state.moves.forEach((a, i) => a >= CELLS && owners.set(a, i % 2));
  const showMarkers =
    state.hint ?? (state.analysis && ((state.mode === 'explore' && !drilling()) || state.liveAnalysis) ? state.analysis.result : null);
  board.render({
    game: g,
    wallOwners: owners,
    lastMove: state.moves.length ? state.moves[state.moves.length - 1] : null,
    interactive: !g.over && !isAiTurn(g) && !state.thinking && !state.grading && !drillDone(),
    markers: markersFrom(showMarkers, 3),
    flipped: state.flipped,
  });
  renderTray();
}

function renderPanels() {
  renderSetup();
  renderStatus();
  renderCards();
  renderMoves();
  renderCoach();
  renderAnalysis();
  renderEval();
  document.querySelectorAll<HTMLButtonElement>('#modes button').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === state.mode);
    b.setAttribute('aria-selected', String(b.dataset.mode === state.mode));
  });
  const g = currentGame();
  ($('btn-undo') as HTMLButtonElement).disabled = state.moves.length === 0;
  ($('btn-redo') as HTMLButtonElement).disabled = state.redo.length === 0;
  ($('btn-hint') as HTMLButtonElement).disabled = g.over || isAiTurn(g);
  ($('chk-analysis') as HTMLInputElement).checked = state.liveAnalysis || state.mode === 'explore';
  ($('chk-analysis') as HTMLInputElement).disabled = state.mode === 'explore';
  ($('chk-analysis').parentElement as HTMLElement).hidden = drilling();
  ($('btn-redo') as HTMLButtonElement).hidden = drilling();
}

const MODE_INTRO: Record<Mode, string> = {
  ai: 'Play the trained engine. Pick a side and a strength.',
  local: 'Pass-and-play on one screen. Ask for a hint whenever you like.',
  coach: 'Play the engine while it grades every move you make — with extra attention to your opening.',
  explore: 'Study openings. <b>Explore</b>: play both sides while the engine ranks every candidate (click one to play it). <b>Drill</b>: find good moves yourself against varied book replies.',
};

function renderSetup() {
  const el = $('setup');
  const vsAi = state.mode === 'ai' || state.mode === 'coach';
  el.innerHTML = `
    <p class="intro">${MODE_INTRO[state.mode]}</p>
    ${
      vsAi
        ? `<div class="field"><span>You play</span><div class="seg" id="side-seg">
            ${(['white', 'black', 'random'] as const)
              .map((s) => `<button data-side="${s}" class="${state.sidePref === s ? 'active' : ''}">${s[0].toUpperCase() + s.slice(1)}</button>`)
              .join('')}
          </div></div>
          <div class="field"><span>Strength</span><div class="levels" id="level-seg">
            ${LEVELS.map((l) => `<button data-level="${l.id}" class="${state.level === l.id ? 'active' : ''}" title="${l.blurb}">${l.id}<small>${l.name}</small></button>`).join('')}
          </div></div>
          <p class="muted small">${levelById(state.level).blurb} Side changes apply on New game.</p>`
        : ''
    }
    ${
      state.mode === 'explore'
        ? `<div class="seg" id="explore-seg">
            <button data-drill="off" class="${!state.drillOn ? 'active' : ''}">Explore</button>
            <button data-drill="0" class="${state.drillOn && state.drillSide === 0 ? 'active' : ''}">Drill as White</button>
            <button data-drill="1" class="${state.drillOn && state.drillSide === 1 ? 'active' : ''}">Drill as Black</button>
          </div>`
        : ''
    }`;
  el.querySelectorAll<HTMLButtonElement>('[data-drill]').forEach((b) =>
    b.addEventListener('click', () => {
      const v = b.dataset.drill!;
      state.drillOn = v !== 'off';
      if (state.drillOn) state.drillSide = Number(v);
      newGame();
    }),
  );
  el.querySelectorAll<HTMLButtonElement>('[data-side]').forEach((b) =>
    b.addEventListener('click', () => {
      state.sidePref = b.dataset.side as 'white' | 'black' | 'random';
      saveSettings();
      renderSetup();
    }),
  );
  el.querySelectorAll<HTMLButtonElement>('[data-level]').forEach((b) =>
    b.addEventListener('click', () => {
      state.level = Number(b.dataset.level);
      saveSettings();
      renderSetup();
    }),
  );
}

function renderStatus() {
  const g = currentGame();
  const s = $('status');
  if (g.over) {
    if (g.winner === 2) s.innerHTML = `<b>Draw</b> — move limit reached.`;
    else {
      const vsAi = state.mode === 'ai' || state.mode === 'coach';
      const msg = vsAi ? (g.winner === state.humanSide ? 'You win! 🎉' : 'The computer wins.') : `${playerName(g.winner)} wins!`;
      s.innerHTML = `<b>${msg}</b> <span class="muted">${g.ply} moves</span>`;
    }
    return;
  }
  const who = playerName(g.turn);
  let text = `<span class="dot p${g.turn}"></span> ${who} to move`;
  if (state.thinking) text = `<span class="dot p${g.turn}"></span> Computer is thinking<span class="ellipsis"></span>`;
  else if (state.mode === 'ai' || state.mode === 'coach') text = `<span class="dot p${g.turn}"></span> Your move (${who})`;
  s.innerHTML = text;
}

function renderCards() {
  const g = currentGame();
  const vsAi = state.mode === 'ai' || state.mode === 'coach';
  const label = (p: number) => {
    if (!vsAi) return playerName(p);
    return p === state.humanSide ? `You · ${playerName(p)}` : `Computer · ${levelById(state.level).name}`;
  };
  const card = (p: number) => {
    const walls = Array.from({ length: 10 }, (_, i) => `<i class="${i < g.walls[p] ? 'on' : ''}"></i>`).join('');
    return `<span class="dot p${p}"></span><span class="pname">${label(p)}</span>
      <span class="dist" title="Shortest path to goal">${g.distance(p)} to go</span>
      <span class="wallbank" title="${g.walls[p]} walls left">${walls}</span>`;
  };
  const top = state.flipped ? 0 : 1;
  $('card-top').innerHTML = card(top);
  $('card-bottom').innerHTML = card(1 - top);
  $('card-top').classList.toggle('active', g.turn === top && !g.over);
  $('card-bottom').classList.toggle('active', g.turn !== top && !g.over);
}

function renderMoves() {
  const ol = $('movelist');
  const items: string[] = [];
  for (let i = 0; i < state.moves.length; i += 2) {
    const cellFor = (ply: number) => {
      if (ply >= state.moves.length) return '<span></span>';
      const fb = state.feedback.get(ply);
      const sym = fb ? GRADE_INFO[fb.grade].symbol : '';
      const cls = fb ? `g-${fb.grade}` : '';
      const last = ply === state.moves.length - 1 ? ' last' : '';
      return `<span class="mv ${cls}${last}">${actionToString(state.moves[ply])}${sym ? `<sup>${sym}</sup>` : ''}</span>`;
    };
    items.push(`<li><span class="num">${i / 2 + 1}.</span>${cellFor(i)}${cellFor(i + 1)}</li>`);
  }
  ol.innerHTML = items.join('') || '<li class="muted empty">No moves yet</li>';
  ol.scrollTop = ol.scrollHeight;
}

function renderCoach() {
  const panel = $('coach-panel');
  if (drilling()) return renderDrill(panel);
  if (state.mode !== 'coach') {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const mine = [...state.feedback.values()].sort((a, b) => a.ply - b.ply);
  const latest = mine[mine.length - 1];
  const opening = mine.filter((f) => f.ply < OPENING_PLIES);
  const acc = (fs: Feedback[]) => (fs.length ? Math.round(100 * (1 - fs.reduce((s, f) => s + Math.min(f.drop, 0.5), 0) / fs.length / 0.5)) : null);
  const openAcc = acc(opening);
  const allAcc = acc(mine);
  let body = '';
  if (state.grading) body = `<div class="fb pending">Reviewing your move<span class="ellipsis"></span></div>`;
  else if (latest) {
    const info = GRADE_INFO[latest.grade];
    const better =
      latest.grade === 'best'
        ? `That's the engine's top choice.`
        : `Engine preferred <b>${actionToString(latest.bestAction)}</b> (${Math.round(latest.bestWin * 100)}% vs your ${Math.round(latest.playedWin * 100)}%).<br><span class="muted">Line: ${fmtPv(latest.bestPv)}</span>`;
    const phase = latest.ply < OPENING_PLIES ? '<span class="tag">Opening</span>' : '';
    body = `<div class="fb g-${latest.grade}"><div class="fb-head">${phase}<b>${actionToString(latest.action)}</b> — ${info.label} ${info.symbol}</div><div>${better}</div></div>`;
  } else body = `<div class="fb muted">Make a move and I'll tell you how it stacks up.</div>`;
  const counts = (['blunder', 'mistake', 'inaccuracy'] as Grade[])
    .map((gr) => ({ gr, n: mine.filter((f) => f.grade === gr).length }))
    .filter((x) => x.n > 0)
    .map((x) => `${x.n} ${GRADE_INFO[x.gr].label.toLowerCase()}${x.n > 1 ? 's' : ''}`)
    .join(' · ');
  panel.innerHTML = `<h2>Coach</h2>${body}
    <div class="stats">
      <div><span class="big">${openAcc ?? '—'}${openAcc !== null ? '%' : ''}</span><span class="muted">opening accuracy</span></div>
      <div><span class="big">${allAcc ?? '—'}${allAcc !== null ? '%' : ''}</span><span class="muted">overall accuracy</span></div>
    </div>
    ${counts ? `<p class="muted small">${counts}</p>` : ''}`;
}

function renderDrill(panel: HTMLElement) {
  panel.hidden = false;
  const d = state.drill;
  const pct = d.total ? Math.round((100 * d.right) / d.total) : null;
  const yourMoves = Math.ceil(DRILL_PLIES / 2);
  let body: string;
  if (state.grading) body = `<div class="fb pending">Checking<span class="ellipsis"></span></div>`;
  else if (drillDone())
    body = `<div class="fb g-best"><b>Line complete!</b> You found a sound move first try ${d.right} of ${d.total} times.</div>
      <button class="primary" id="btn-next-line">Next line</button>`;
  else if (d.msg) body = `<div class="fb ${d.cls}">${d.msg}</div>`;
  else body = `<div class="fb muted">Find a strong move for ${playerName(state.drillSide)}. The engine replies with varied book moves.</div>`;
  panel.innerHTML = `<h2>Opening drill</h2>${body}
    <div class="stats">
      <div><span class="big">${d.right}/${d.total}</span><span class="muted">first-try correct</span></div>
      <div><span class="big">${pct ?? '—'}${pct !== null ? '%' : ''}</span><span class="muted">of ${yourMoves} moves per line</span></div>
    </div>`;
  panel.querySelector('#btn-next-line')?.addEventListener('click', newGame);
}

function renderAnalysis() {
  const panel = $('analysis-panel');
  const show = state.hint || (state.mode === 'explore' && !drilling()) || state.liveAnalysis;
  const res = state.hint ?? state.analysis?.result ?? null;
  const g = currentGame();
  if (!show || g.over) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const src = state.analysis?.source === 'book' ? '<span class="tag book">Opening book</span>' : '';
  const progress = state.analysis && !state.analysis.done ? `<span class="muted small">searching… ${res?.visits ?? 0}</span>` : res ? `<span class="muted small">${res.visits.toLocaleString()} sims</span>` : '';
  const rows = (res?.moves ?? [])
    .filter((m) => m.visits > 0 || (res?.moves.length ?? 0) === 1)
    .slice(0, 6)
    .map((m: MoveStat, i) => {
      const pct = winPct(m.q);
      const share = res && res.visits > 1 ? Math.round((100 * m.visits) / Math.max(1, res.visits - 1)) : 100;
      return `<li class="an-row" data-action="${m.action}" tabindex="0">
        <span class="rank r${Math.min(i, 2)}">${i + 1}</span>
        <span class="an-move">${actionToString(m.action)}</span>
        <span class="an-bar"><span style="width:${pct}%"></span></span>
        <span class="an-pct">${pct}%</span>
        <span class="an-pv muted">${fmtPv(m.pv.slice(1))}</span>
        <span class="an-share muted" title="share of search">${share}%</span>
      </li>`;
    })
    .join('');
  panel.innerHTML = `<div class="panel-head"><h2>${state.hint ? 'Hint' : 'Analysis'}</h2>${src}${progress}</div>
    <p class="muted small">Win chances for ${playerName(g.turn)} after each move${state.mode === 'explore' ? ' · click to play' : ''}.</p>
    <ol class="analysis">${rows || '<li class="muted">Thinking…</li>'}</ol>`;
  if (state.mode === 'explore' || state.mode === 'local') {
    panel.querySelectorAll<HTMLElement>('.an-row').forEach((row) => {
      const go = () => humanMove(Number(row.dataset.action));
      row.addEventListener('click', go);
      row.addEventListener('keydown', (e) => e.key === 'Enter' && go());
    });
  }
}

function renderEval() {
  const g = currentGame();
  const res = state.hint ?? state.analysis?.result;
  const showEval = ((state.mode === 'explore' && !drilling()) || state.liveAnalysis || state.hint) && res && !g.over;
  const bar = $('evalbar');
  bar.classList.toggle('hidden', !showEval);
  if (!showEval || !res) return;
  const whiteQ = g.turn === 0 ? res.value : -res.value;
  const pct = winPct(whiteQ);
  const fill = $('evalbar-fill');
  // bar shows white from the bottom when not flipped
  fill.style.height = `${state.flipped ? 100 - pct : pct}%`;
  bar.classList.toggle('flipped', state.flipped);
  $('evalbar-label').textContent = `${pct}`;
}

// ---------------- wall tray (drag & drop) ----------------
function setupWallTray() {
  const ghost = document.createElement('div');
  ghost.className = 'wall-ghost';
  ghost.hidden = true;
  document.body.appendChild(ghost);
  document.querySelectorAll<HTMLButtonElement>('.tray-piece').forEach((piece) => {
    const horizontal = piece.dataset.orient === 'h';
    let dragging = false;
    let target: number | null = null;
    // keep the wall visible above a finger; a mouse doesn't cover anything
    let lift = 0;
    const move = (e: PointerEvent) => {
      if (!dragging) return;
      const x = e.clientX;
      const y = e.clientY - lift;
      ghost.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
      target = board.dragWallTo(x, y, horizontal);
      ghost.classList.toggle('over-board', target !== null);
    };
    let pointerId = -1;
    const onMove = (e: PointerEvent) => e.pointerId === pointerId && move(e);
    const onUp = (e: PointerEvent) => e.pointerId === pointerId && finish(e.type === 'pointerup');
    const finish = (commit: boolean) => {
      if (!dragging) return;
      dragging = false;
      ghost.hidden = true;
      board.endWallDrag();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      if (commit && target !== null) humanMove(target);
    };
    piece.addEventListener('pointerdown', (e) => {
      if (!board.canPlaceWalls()) return;
      e.preventDefault();
      pointerId = e.pointerId;
      dragging = true;
      target = null;
      lift = e.pointerType === 'mouse' ? 0 : 56;
      ghost.className = `wall-ghost ${horizontal ? 'h' : 'v'} p${currentGame().turn}`;
      ghost.hidden = false;
      // follow the pointer anywhere on the page, not just over the tray button
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
      window.addEventListener('pointercancel', onUp);
      move(e);
    });
  });
}

function renderTray() {
  const g = currentGame();
  const tray = $('wall-tray');
  const can = board.canPlaceWalls();
  tray.classList.toggle('disabled', !can);
  tray.dataset.player = String(g.turn);
  $('wall-tray').querySelector('.tray-label')!.textContent = g.over
    ? 'Game over'
    : g.walls[g.turn] === 0
      ? `${playerName(g.turn)} has no walls left`
      : can
        ? `Drag a wall onto the board (${g.walls[g.turn]} left)`
        : 'Waiting…';
}

// ---------------- wiring ----------------
setupWallTray();
document.querySelectorAll<HTMLButtonElement>('#modes button').forEach((b) =>
  b.addEventListener('click', () => {
    const m = b.dataset.mode as Mode;
    if (m === state.mode) return;
    state.mode = m;
    newGame();
  }),
);
$('btn-new').addEventListener('click', newGame);
$('btn-undo').addEventListener('click', undo);
$('btn-redo').addEventListener('click', redo);
$('btn-hint').addEventListener('click', hint);
$('btn-flip').addEventListener('click', () => {
  state.flipped = !state.flipped;
  renderBoard();
  renderCards();
  renderEval();
});
$('chk-analysis').addEventListener('change', (e) => {
  state.liveAnalysis = (e.target as HTMLInputElement).checked;
  saveSettings();
  if (!state.liveAnalysis) state.analysis = null;
  refresh();
});
window.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') undo();
  else if (e.key === 'ArrowRight') redo();
  else if (e.key === 'h' || e.key === 'H') hint();
  else if (e.key === 'f' || e.key === 'F') $('btn-flip').click();
});

newGame();
