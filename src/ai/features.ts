import { CELLS, Game, H_OFFSET, N, NUM_ACTIONS, V_OFFSET, WSLOTS } from '../engine/quoridor.ts';

/**
 * Network input encoding, always from the perspective of the player to move.
 * Player 1's view is flipped vertically so the network only ever "plays upward".
 */
export const INPUT_SIZE = CELLS * 4 + WSLOTS * 2 + 5; // 457

const flipCell = (cell: number) => (N - 1 - ((cell / N) | 0)) * N + (cell % N);
const flipWall = (w: number) => (7 - ((w / 8) | 0)) * 8 + (w % 8);
const mirrorCell = (cell: number) => ((cell / N) | 0) * N + (N - 1 - (cell % N));
const mirrorWall = (w: number) => ((w / 8) | 0) * 8 + (7 - (w % 8));

function mapAction(a: number, fc: (c: number) => number, fw: (w: number) => number): number {
  if (a < CELLS) return fc(a);
  if (a < V_OFFSET) return H_OFFSET + fw(a - H_OFFSET);
  return V_OFFSET + fw(a - V_OFFSET);
}

/** Vertical flip of an action (an involution). Used for player-1 perspective. */
export const flipAction = (a: number) => mapAction(a, flipCell, flipWall);
/** Left-right mirror of an action (an involution). Used for data augmentation. */
export const mirrorAction = (a: number) => mapAction(a, mirrorCell, mirrorWall);

/** Board action -> canonical (side-to-move) action index. */
export const toCanonical = (a: number, turn: number) => (turn === 0 ? a : flipAction(a));
export const fromCanonical = toCanonical;

/** Precomputed permutation tables. */
export const FLIP_PERM = Int16Array.from({ length: NUM_ACTIONS }, (_, a) => flipAction(a));
export const MIRROR_PERM = Int16Array.from({ length: NUM_ACTIONS }, (_, a) => mirrorAction(a));

const field0 = new Uint8Array(CELLS);
const field1 = new Uint8Array(CELLS);

export function encode(g: Game, out: Float32Array = new Float32Array(INPUT_SIZE)): Float32Array {
  out.fill(0);
  const me = g.turn;
  const opp = 1 - me;
  const flip = me === 1;
  const fc = flip ? flipCell : (c: number) => c;
  const fw = flip ? flipWall : (w: number) => w;

  out[fc(g.pos[me])] = 1;
  out[CELLS + fc(g.pos[opp])] = 1;
  const hb = CELLS * 2;
  const vb = hb + WSLOTS;
  for (let w = 0; w < WSLOTS; w++) {
    if (g.hw[w]) out[hb + fw(w)] = 1;
    if (g.vw[w]) out[vb + fw(w)] = 1;
  }
  const myField = g.distanceField(me, field0);
  const oppField = g.distanceField(opp, field1);
  const db = vb + WSLOTS;
  for (let c = 0; c < CELLS; c++) {
    const m = myField[c];
    const o = oppField[c];
    out[db + fc(c)] = m === 255 ? 2 : m / 16;
    out[db + CELLS + fc(c)] = o === 255 ? 2 : o / 16;
  }
  const sb = db + CELLS * 2;
  const myDist = myField[g.pos[me]];
  const oppDist = oppField[g.pos[opp]];
  out[sb] = g.walls[me] / 10;
  out[sb + 1] = g.walls[opp] / 10;
  out[sb + 2] = myDist / 16;
  out[sb + 3] = oppDist / 16;
  out[sb + 4] = (oppDist - myDist) / 8;
  return out;
}

/** Mirror an encoded input left-right (for augmentation). */
export function mirrorInput(x: Float32Array, out: Float32Array = new Float32Array(INPUT_SIZE)): Float32Array {
  for (let c = 0; c < CELLS; c++) {
    const m = mirrorCell(c);
    out[m] = x[c];
    out[CELLS + m] = x[CELLS + c];
    const db = CELLS * 2 + WSLOTS * 2;
    out[db + m] = x[db + c];
    out[db + CELLS + m] = x[db + CELLS + c];
  }
  const hb = CELLS * 2;
  const vb = hb + WSLOTS;
  for (let w = 0; w < WSLOTS; w++) {
    const m = mirrorWall(w);
    out[hb + m] = x[hb + w];
    out[vb + m] = x[vb + w];
  }
  const sb = INPUT_SIZE - 5;
  for (let i = 0; i < 5; i++) out[sb + i] = x[sb + i];
  return out;
}
