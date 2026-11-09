import { CELLS, Game, H_OFFSET, N, V_OFFSET } from '../engine/quoridor.ts';

const SVG_NS = 'http://www.w3.org/2000/svg';
const CELL = 54;
const GAP = 14;
const PITCH = CELL + GAP;
const PAD = 26;
const SIZE = PAD * 2 + N * CELL + (N - 1) * GAP;
const FILES = 'abcdefghi';

export interface Marker {
  action: number;
  label: string;
  /** 0 = best, higher = weaker; drives colour */
  rank: number;
}

export interface BoardView {
  game: Game;
  /** owner (0/1) of each placed wall action, for colouring */
  wallOwners: Map<number, number>;
  lastMove: number | null;
  /** can the user interact right now? */
  interactive: boolean;
  markers: Marker[];
  flipped: boolean;
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.appendChild(e);
  return e;
}

export class BoardUI {
  readonly svg: SVGSVGElement;
  private layerCells: SVGGElement;
  private layerWalls: SVGGElement;
  private layerMarkers: SVGGElement;
  private layerPawns: SVGGElement;
  private layerPreview: SVGGElement;
  private view: BoardView | null = null;
  private legalPawn = new Set<number>();
  private legalWalls: Set<number> | null = null;
  private preview: number | null = null;
  onAction: (a: number) => void = () => {};

  constructor(container: HTMLElement) {
    this.svg = el('svg', { viewBox: `0 0 ${SIZE} ${SIZE}`, class: 'board', role: 'img', 'aria-label': 'Quoridor board' });
    el('rect', { x: 0, y: 0, width: SIZE, height: SIZE, rx: 18, class: 'board-bg' }, this.svg);
    this.layerCells = el('g', {}, this.svg);
    this.layerWalls = el('g', {}, this.svg);
    this.layerMarkers = el('g', {}, this.svg);
    this.layerPawns = el('g', {}, this.svg);
    this.layerPreview = el('g', { 'pointer-events': 'none' }, this.svg);
    container.appendChild(this.svg);

    this.svg.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'touch') return;
      const cell = this.cellAt(e);
      this.setPreview(cell >= 0 && this.legalPawn.has(cell) ? null : this.wallAt(e));
    });
    this.svg.addEventListener('pointerleave', () => this.setPreview(null));
    this.svg.addEventListener('pointerdown', (e) => this.handleDown(e));
  }

  // ---------- geometry ----------
  private dc(c: number) {
    return this.view?.flipped ? N - 1 - c : c;
  }
  private dr(r: number) {
    return this.view?.flipped ? r : N - 1 - r;
  }
  private cx(c: number) {
    return PAD + this.dc(c) * PITCH + CELL / 2;
  }
  private cy(r: number) {
    return PAD + this.dr(r) * PITCH + CELL / 2;
  }

  private wallRect(a: number) {
    const horizontal = a < V_OFFSET;
    const w = horizontal ? a - H_OFFSET : a - V_OFFSET;
    const r = (w / 8) | 0;
    const c = w % 8;
    const x = (this.cx(c) + this.cx(c + 1)) / 2;
    const y = (this.cy(r) + this.cy(r + 1)) / 2;
    const long = 2 * CELL + GAP - 4;
    const thick = GAP - 3;
    return horizontal
      ? { x: x - long / 2, y: y - thick / 2, width: long, height: thick }
      : { x: x - thick / 2, y: y - long / 2, width: thick, height: long };
  }

  private toSvg(e: PointerEvent) {
    const pt = this.svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    return pt.matrixTransform(this.svg.getScreenCTM()!.inverse());
  }

  /** Board cell under pointer, or -1 */
  private cellAt(e: PointerEvent): number {
    const p = this.toSvg(e);
    const gx = p.x - PAD;
    const gy = p.y - PAD;
    const dcol = Math.floor(gx / PITCH);
    const drow = Math.floor(gy / PITCH);
    if (dcol < 0 || dcol >= N || drow < 0 || drow >= N) return -1;
    if (gx - dcol * PITCH > CELL + 2 || gy - drow * PITCH > CELL + 2) return -1;
    const c = this.view?.flipped ? N - 1 - dcol : dcol;
    const r = this.view?.flipped ? drow : N - 1 - drow;
    return r * N + c;
  }

  /** Wall action under the pointer (snapped), or null if the pointer is well inside a cell. */
  private wallAt(e: PointerEvent): number | null {
    if (!this.view?.interactive || !this.legalWalls) return null;
    const p = this.toSvg(e);
    const gx = p.x - PAD;
    const gy = p.y - PAD;
    // fingers are fat: let touches land a little way into a square and still count as a groove
    const tol = e.pointerType === 'touch' ? 13 : 5;
    const gapX = Math.round((gx - CELL - GAP / 2) / PITCH);
    const gapY = Math.round((gy - CELL - GAP / 2) / PITCH);
    const dx = Math.abs(gx - (gapX * PITCH + CELL + GAP / 2));
    const dy = Math.abs(gy - (gapY * PITCH + CELL + GAP / 2));
    const inV = gapX >= 0 && gapX <= 7 && dx <= GAP / 2 + tol;
    const inH = gapY >= 0 && gapY <= 7 && dy <= GAP / 2 + tol;
    if (!inV && !inH) return null;
    const horizontal = inH && (!inV || dy <= dx);
    let dCol: number;
    let dRow: number;
    if (horizontal) {
      dRow = gapY;
      dCol = Math.floor((gx - CELL / 2) / PITCH);
    } else {
      dCol = gapX;
      dRow = Math.floor((gy - CELL / 2) / PITCH);
    }
    if (dCol < -1 || dCol > 8 || dRow < -1 || dRow > 8) return null;
    dCol = Math.max(0, Math.min(7, dCol));
    dRow = Math.max(0, Math.min(7, dRow));
    // display corner (dRow,dCol) sits between display rows dRow,dRow+1 and cols dCol,dCol+1
    const c = this.view.flipped ? 7 - dCol : dCol;
    const r = this.view.flipped ? dRow : 7 - dRow;
    return (horizontal ? H_OFFSET : V_OFFSET) + r * 8 + c;
  }

  private handleDown(e: PointerEvent) {
    if (!this.view?.interactive) return;
    const cell = this.cellAt(e);
    if (cell >= 0 && this.legalPawn.has(cell)) {
      this.setPreview(null);
      this.onAction(cell);
      return;
    }
    const wall = this.wallAt(e);
    if (wall === null) return;
    if (!this.legalWalls?.has(wall)) {
      this.flashIllegal(wall);
      return;
    }
    if (e.pointerType === 'touch' && this.preview !== wall) {
      // first tap previews, second tap confirms
      this.setPreview(wall);
      return;
    }
    this.setPreview(null);
    this.onAction(wall);
  }

  private flashIllegal(a: number) {
    const r = this.wallRect(a);
    const rect = el('rect', { ...r, rx: 3, class: 'wall-illegal' }, this.layerPreview);
    setTimeout(() => rect.remove(), 350);
  }

  private setPreview(a: number | null) {
    if (a === this.preview) return;
    this.preview = a;
    this.layerPreview.replaceChildren();
    if (a === null || !this.view) return;
    const legal = this.legalWalls?.has(a);
    el('rect', { ...this.wallRect(a), rx: 3, class: legal ? `wall-preview p${this.view.game.turn}` : 'wall-illegal' }, this.layerPreview);
  }

  // ---------- rendering ----------
  render(view: BoardView) {
    this.view = view;
    const g = view.game;
    this.legalPawn = new Set(view.interactive ? g.pawnMoves() : []);
    this.legalWalls = view.interactive && g.walls[g.turn] > 0 ? new Set(g.wallMoves()) : null;
    this.preview = null;
    this.layerPreview.replaceChildren();

    // cells
    this.layerCells.replaceChildren();
    for (let cell = 0; cell < CELLS; cell++) {
      const r = (cell / N) | 0;
      const c = cell % N;
      let cls = 'cell';
      if (r === N - 1) cls += ' goal0';
      if (r === 0) cls += ' goal1';
      if (view.lastMove === cell) cls += ' last';
      el('rect', { x: this.cx(c) - CELL / 2, y: this.cy(r) - CELL / 2, width: CELL, height: CELL, rx: 7, class: cls }, this.layerCells);
      if (this.legalPawn.has(cell)) el('circle', { cx: this.cx(c), cy: this.cy(r), r: 8, class: `target p${g.turn}` }, this.layerCells);
    }
    // coordinates
    for (let c = 0; c < N; c++) {
      const t = el('text', { x: this.cx(c), y: SIZE - 8, class: 'coord' }, this.layerCells);
      t.textContent = FILES[c];
    }
    for (let r = 0; r < N; r++) {
      const t = el('text', { x: 11, y: this.cy(r) + 4, class: 'coord' }, this.layerCells);
      t.textContent = String(r + 1);
    }

    // walls
    this.layerWalls.replaceChildren();
    for (let w = 0; w < 64; w++) {
      for (const [arr, off] of [
        [g.hw, H_OFFSET],
        [g.vw, V_OFFSET],
      ] as const) {
        if (!arr[w]) continue;
        const a = off + w;
        const owner = view.wallOwners.get(a) ?? 0;
        const cls = `wall p${owner}${view.lastMove === a ? ' last' : ''}`;
        el('rect', { ...this.wallRect(a), rx: 3, class: cls }, this.layerWalls);
      }
    }

    // markers (hints / analysis)
    this.layerMarkers.replaceChildren();
    for (const m of view.markers) {
      const cls = `marker rank${Math.min(m.rank, 2)}`;
      let lx: number;
      let ly: number;
      if (m.action < CELLS) {
        const r = (m.action / N) | 0;
        const c = m.action % N;
        lx = this.cx(c);
        ly = this.cy(r);
        el('circle', { cx: lx, cy: ly, r: CELL / 2 - 5, class: cls + ' ring' }, this.layerMarkers);
      } else {
        const rect = this.wallRect(m.action);
        el('rect', { ...rect, rx: 3, class: cls + ' ghost' }, this.layerMarkers);
        lx = rect.x + rect.width / 2;
        ly = rect.y + rect.height / 2;
      }
      el('circle', { cx: lx, cy: ly, r: 10, class: cls + ' badge' }, this.layerMarkers);
      const t = el('text', { x: lx, y: ly + 4, class: 'badge-text' }, this.layerMarkers);
      t.textContent = m.label;
    }

    // pawns
    this.layerPawns.replaceChildren();
    for (let p = 0; p < 2; p++) {
      const cell = g.pos[p];
      const r = (cell / N) | 0;
      const c = cell % N;
      const grp = el('g', { class: `pawn p${p}${g.turn === p && !g.over ? ' to-move' : ''}` }, this.layerPawns);
      el('circle', { cx: this.cx(c), cy: this.cy(r) + 2, r: CELL / 2 - 9, class: 'pawn-shadow' }, grp);
      el('circle', { cx: this.cx(c), cy: this.cy(r), r: CELL / 2 - 9, class: 'pawn-body' }, grp);
      el('circle', { cx: this.cx(c) - 5, cy: this.cy(r) - 6, r: 5, class: 'pawn-shine' }, grp);
    }
  }
}
