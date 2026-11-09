/**
 * Quoridor rules engine (2 players, 9x9 board, 10 walls each).
 *
 * Coordinates
 *   cell  = row * 9 + col, row 0 is rank "1" (player 0's home row), col 0 is file "a".
 *   wall  = r * 8 + c  (r, c in 0..7) — the wall's centre sits on the corner
 *           shared by cells (r,c), (r,c+1), (r+1,c), (r+1,c+1).
 *     horizontal: between rows r and r+1, covering cols c and c+1.
 *     vertical:   between cols c and c+1, covering rows r and r+1.
 *
 * Actions (209 total)
 *   0..80     move pawn to cell
 *   81..144   horizontal wall  (81 + w)
 *   145..208  vertical wall    (145 + w)
 *
 * Player 0 starts on e1 and races to row 8; player 1 starts on e9 and races to row 0.
 */

export const N = 9;
export const CELLS = 81;
export const WSLOTS = 64;
export const NUM_ACTIONS = CELLS + 2 * WSLOTS; // 209
export const H_OFFSET = CELLS;
export const V_OFFSET = CELLS + WSLOTS;
export const WALLS_PER_PLAYER = 10;
export const DEFAULT_MAX_PLIES = 250;

export const rowOf = (cell: number) => (cell / N) | 0;
export const colOf = (cell: number) => cell % N;
export const goalRow = (player: number) => (player === 0 ? N - 1 : 0);

export const isPawnAction = (a: number) => a < CELLS;
export const isHWall = (a: number) => a >= H_OFFSET && a < V_OFFSET;
export const isVWall = (a: number) => a >= V_OFFSET;

// Direction indices: 0 = up (+row), 1 = down, 2 = right (+col), 3 = left
const DR = [1, -1, 0, 0];
const DC = [0, 0, 1, -1];
const PERP: number[][] = [[2, 3], [2, 3], [0, 1], [0, 1]];

export class Game {
  /** pawn cell for each player */
  pos: Int16Array = new Int16Array(2);
  /** walls remaining for each player */
  walls: Int8Array = new Int8Array(2);
  /** player to move */
  turn = 0;
  ply = 0;
  /** -1 = ongoing, 0/1 = winner, 2 = draw (ply cap) */
  winner = -1;
  maxPlies = DEFAULT_MAX_PLIES;
  hw: Uint8Array = new Uint8Array(WSLOTS);
  vw: Uint8Array = new Uint8Array(WSLOTS);
  /** blockUp[cell] = 1 if moving from cell to cell+9 is blocked by a wall */
  blockUp: Uint8Array = new Uint8Array(CELLS);
  /** blockRight[cell] = 1 if moving from cell to cell+1 is blocked by a wall */
  blockRight: Uint8Array = new Uint8Array(CELLS);

  constructor(init = true) {
    if (init) {
      this.pos[0] = 4; // e1
      this.pos[1] = 8 * N + 4; // e9
      this.walls[0] = WALLS_PER_PLAYER;
      this.walls[1] = WALLS_PER_PLAYER;
    }
  }

  clone(): Game {
    const g = new Game(false);
    g.pos.set(this.pos);
    g.walls.set(this.walls);
    g.turn = this.turn;
    g.ply = this.ply;
    g.winner = this.winner;
    g.maxPlies = this.maxPlies;
    g.hw.set(this.hw);
    g.vw.set(this.vw);
    g.blockUp.set(this.blockUp);
    g.blockRight.set(this.blockRight);
    return g;
  }

  get over(): boolean {
    return this.winner !== -1;
  }

  /** Can a piece step from `cell` in direction d (ignoring pawns)? Returns target or -1. */
  step(cell: number, d: number): number {
    const r = rowOf(cell);
    const c = colOf(cell);
    switch (d) {
      case 0:
        return r < N - 1 && !this.blockUp[cell] ? cell + N : -1;
      case 1:
        return r > 0 && !this.blockUp[cell - N] ? cell - N : -1;
      case 2:
        return c < N - 1 && !this.blockRight[cell] ? cell + 1 : -1;
      default:
        return c > 0 && !this.blockRight[cell - 1] ? cell - 1 : -1;
    }
  }

  /** Legal pawn destinations for the player to move. */
  pawnMoves(player = this.turn): number[] {
    const me = this.pos[player];
    const opp = this.pos[1 - player];
    const out: number[] = [];
    for (let d = 0; d < 4; d++) {
      const n = this.step(me, d);
      if (n < 0) continue;
      if (n !== opp) {
        out.push(n);
        continue;
      }
      const straight = this.step(opp, d);
      if (straight >= 0) {
        out.push(straight);
        continue;
      }
      for (const pd of PERP[d]) {
        const diag = this.step(opp, pd);
        if (diag >= 0 && diag !== me && !out.includes(diag)) out.push(diag);
      }
    }
    return out;
  }

  /** Wall placement is geometrically possible (no overlap / crossing); ignores path rule and wall count. */
  wallFits(horizontal: boolean, w: number): boolean {
    const c = w % 8;
    if (horizontal) {
      if (this.hw[w] || this.vw[w]) return false;
      if (c > 0 && this.hw[w - 1]) return false;
      if (c < 7 && this.hw[w + 1]) return false;
      return true;
    }
    if (this.vw[w] || this.hw[w]) return false;
    if (w >= 8 && this.vw[w - 8]) return false;
    if (w < 56 && this.vw[w + 8]) return false;
    return true;
  }

  setWall(horizontal: boolean, w: number, v: 0 | 1) {
    const r = (w / 8) | 0;
    const c = w % 8;
    const base = r * N + c;
    if (horizontal) {
      this.hw[w] = v;
      this.blockUp[base] = v;
      this.blockUp[base + 1] = v;
    } else {
      this.vw[w] = v;
      this.blockRight[base] = v;
      this.blockRight[base + N] = v;
    }
  }

  /**
   * BFS distance field to `player`'s goal row, ignoring pawns.
   * Unreachable cells get 255. Writes into `out` if given.
   */
  distanceField(player: number, out: Uint8Array = new Uint8Array(CELLS)): Uint8Array {
    out.fill(255);
    const q = BFS_QUEUE;
    let head = 0;
    let tail = 0;
    const gr = goalRow(player);
    for (let c = 0; c < N; c++) {
      const cell = gr * N + c;
      out[cell] = 0;
      q[tail++] = cell;
    }
    while (head < tail) {
      const cell = q[head++];
      const nd = out[cell] + 1;
      for (let d = 0; d < 4; d++) {
        const n = this.step(cell, d);
        if (n >= 0 && out[n] === 255) {
          out[n] = nd;
          q[tail++] = n;
        }
      }
    }
    return out;
  }

  /** Shortest path length (in steps, ignoring pawns) for player to reach goal. 255 if none. */
  distance(player: number): number {
    return this.distanceField(player, SCRATCH_FIELD)[this.pos[player]];
  }

  /** Is there any path for player to its goal? (cheap DFS) */
  hasPath(player: number): boolean {
    const start = this.pos[player];
    const gr = goalRow(player);
    const seen = SEEN;
    seen.fill(0);
    const stack = BFS_QUEUE;
    let sp = 0;
    stack[sp++] = start;
    seen[start] = 1;
    // Prefer heading toward goal first for speed
    const order = player === 0 ? [0, 2, 3, 1] : [1, 2, 3, 0];
    while (sp > 0) {
      const cell = stack[--sp];
      if (rowOf(cell) === gr) return true;
      for (let i = 3; i >= 0; i--) {
        const n = this.step(cell, order[i]);
        if (n >= 0 && !seen[n]) {
          seen[n] = 1;
          stack[sp++] = n;
        }
      }
    }
    return false;
  }

  /**
   * Legal wall placements for the player to move (respecting wall count, overlap and path rules).
   * Uses the shortest-path trick: a wall that blocks no edge on a current shortest path of either
   * player cannot disconnect them, so only those need an explicit path check.
   */
  wallMoves(player = this.turn): number[] {
    const out: number[] = [];
    if (this.walls[player] <= 0) return out;
    const p0 = this.pathEdges(0);
    const p1 = this.pathEdges(1);
    for (let h = 0; h < 2; h++) {
      const horizontal = h === 0;
      for (let w = 0; w < WSLOTS; w++) {
        if (!this.wallFits(horizontal, w)) continue;
        const touches = wallTouches(horizontal, w, p0) || wallTouches(horizontal, w, p1);
        if (touches) {
          this.setWall(horizontal, w, 1);
          const ok = this.hasPath(0) && this.hasPath(1);
          this.setWall(horizontal, w, 0);
          if (!ok) continue;
        }
        out.push((horizontal ? H_OFFSET : V_OFFSET) + w);
      }
    }
    return out;
  }

  /** Edges (encoded as up-edge cell or right-edge cell) on one shortest path of `player`. */
  pathEdges(player: number): PathEdges {
    const field = this.distanceField(player, PATH_FIELD[player]);
    const up = PATH_UP[player];
    const right = PATH_RIGHT[player];
    up.fill(0);
    right.fill(0);
    let cell = this.pos[player];
    if (field[cell] === 255) return { up, right };
    while (field[cell] > 0) {
      const want = field[cell] - 1;
      let moved = false;
      for (let d = 0; d < 4; d++) {
        const n = this.step(cell, d);
        if (n >= 0 && field[n] === want) {
          if (d === 0) up[cell] = 1;
          else if (d === 1) up[n] = 1;
          else if (d === 2) right[cell] = 1;
          else right[n] = 1;
          cell = n;
          moved = true;
          break;
        }
      }
      if (!moved) break;
    }
    return { up, right };
  }

  legalActions(): number[] {
    if (this.over) return [];
    const moves = this.pawnMoves();
    return moves.concat(this.wallMoves());
  }

  /** Full legality check for a single action. */
  isLegal(a: number): boolean {
    if (this.over || a < 0 || a >= NUM_ACTIONS) return false;
    if (a < CELLS) return this.pawnMoves().includes(a);
    if (this.walls[this.turn] <= 0) return false;
    const horizontal = a < V_OFFSET;
    const w = horizontal ? a - H_OFFSET : a - V_OFFSET;
    if (!this.wallFits(horizontal, w)) return false;
    this.setWall(horizontal, w, 1);
    const ok = this.hasPath(0) && this.hasPath(1);
    this.setWall(horizontal, w, 0);
    return ok;
  }

  /** Apply an action assumed legal. */
  play(a: number): void {
    const p = this.turn;
    if (a < CELLS) {
      this.pos[p] = a;
      if (rowOf(a) === goalRow(p)) this.winner = p;
    } else {
      const horizontal = a < V_OFFSET;
      this.setWall(horizontal, horizontal ? a - H_OFFSET : a - V_OFFSET, 1);
      this.walls[p]--;
    }
    this.turn = 1 - p;
    this.ply++;
    if (this.winner === -1 && this.ply >= this.maxPlies) this.winner = 2;
  }

  /** Compact string key for a position (for opening books / transpositions). */
  key(): string {
    let s = `${this.turn}${String.fromCharCode(48 + this.pos[0], 48 + this.pos[1])}${this.walls[0]},${this.walls[1]}:`;
    for (let w = 0; w < WSLOTS; w++) {
      if (this.hw[w]) s += 'h' + w.toString(36).padStart(2, '0');
      if (this.vw[w]) s += 'v' + w.toString(36).padStart(2, '0');
    }
    return s;
  }
}

export interface PathEdges {
  up: Uint8Array;
  right: Uint8Array;
}

export function wallTouches(horizontal: boolean, w: number, p: PathEdges): boolean {
  const r = (w / 8) | 0;
  const c = w % 8;
  const base = r * N + c;
  if (horizontal) return p.up[base] === 1 || p.up[base + 1] === 1;
  return p.right[base] === 1 || p.right[base + N] === 1;
}

// Shared scratch buffers (engine is single-threaded per worker)
const BFS_QUEUE = new Int16Array(CELLS * 4);
const SEEN = new Uint8Array(CELLS);
const SCRATCH_FIELD = new Uint8Array(CELLS);
const PATH_FIELD = [new Uint8Array(CELLS), new Uint8Array(CELLS)];
const PATH_UP = [new Uint8Array(CELLS), new Uint8Array(CELLS)];
const PATH_RIGHT = [new Uint8Array(CELLS), new Uint8Array(CELLS)];
