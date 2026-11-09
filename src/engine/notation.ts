import { CELLS, H_OFFSET, N, V_OFFSET, colOf, rowOf } from './quoridor.ts';

/**
 * Standard-ish Quoridor notation:
 *   pawn move  -> destination square, e.g. "e2"
 *   wall       -> square south-west of the wall's centre + orientation, e.g. "e3h", "d6v"
 * Files a–i left to right, ranks 1–9 from player 1's (white's) side.
 */
const FILES = 'abcdefghi';

export function cellName(cell: number): string {
  return FILES[colOf(cell)] + (rowOf(cell) + 1);
}

export function actionToString(a: number): string {
  if (a < CELLS) return cellName(a);
  const horizontal = a < V_OFFSET;
  const w = horizontal ? a - H_OFFSET : a - V_OFFSET;
  const r = (w / 8) | 0;
  const c = w % 8;
  return FILES[c] + (r + 1) + (horizontal ? 'h' : 'v');
}

export function parseAction(s: string): number {
  const m = /^([a-i])([1-9])([hv])?$/.exec(s.trim().toLowerCase());
  if (!m) return -1;
  const c = FILES.indexOf(m[1]);
  const r = Number(m[2]) - 1;
  if (!m[3]) return r * N + c;
  if (r > 7 || c > 7) return -1;
  return (m[3] === 'h' ? H_OFFSET : V_OFFSET) + r * 8 + c;
}
