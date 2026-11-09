import { describe, expect, it } from 'vitest';
import { Game, NUM_ACTIONS } from '../src/engine/quoridor.ts';
import { actionToString, parseAction } from '../src/engine/notation.ts';

const P = (s: string) => {
  const a = parseAction(s);
  if (a < 0) throw new Error('bad ' + s);
  return a;
};
const moves = (g: Game) => g.pawnMoves().map(actionToString).sort();

function playAll(g: Game, seq: string[]) {
  for (const s of seq) {
    const a = P(s);
    expect(g.isLegal(a), `${s} should be legal`).toBe(true);
    g.play(a);
  }
}

describe('notation', () => {
  it('round-trips every action', () => {
    for (let a = 0; a < NUM_ACTIONS; a++) expect(parseAction(actionToString(a))).toBe(a);
  });
});

describe('rules', () => {
  it('starting position', () => {
    const g = new Game();
    expect(moves(g)).toEqual(['d1', 'e2', 'f1']);
    expect(g.wallMoves().length).toBe(128);
    expect(g.distance(0)).toBe(8);
    expect(g.distance(1)).toBe(8);
  });

  it('straight jump', () => {
    const g = new Game();
    playAll(g, ['e2', 'e8', 'e3', 'e7', 'e4', 'e6', 'e5']);
    // black on e6, white on e5; black to move can jump to e4
    expect(moves(g)).toEqual(['d6', 'e4', 'e7', 'f6']);
  });

  it('diagonal jump when wall behind', () => {
    const g = new Game();
    playAll(g, ['e2', 'e8', 'e3', 'e7', 'e4', 'e6', 'e5']);
    // Black places a wall below e5 -> "e4h" sits between rows 4 and 5 covering e,f
    // Instead: put wall so black can't jump straight: wall between e5 and e4 => d4h (covers d,e)
    g.play(P('a1h')); // black wastes a wall elsewhere (turn passes to white)
    g.play(P('d4h')); // white blocks behind itself (between rows 4/5, files d,e)
    // black at e6, white at e5 with wall behind -> diagonals d5, f5
    expect(moves(g)).toEqual(['d5', 'd6', 'e7', 'f5', 'f6']);
  });

  it('walls block movement and overlap rules', () => {
    const g = new Game();
    g.play(P('e1h')); // white: wall between rank1/2 on files e,f
    expect(g.isLegal(P('e1h'))).toBe(false);
    expect(g.isLegal(P('d1h'))).toBe(false); // overlaps
    expect(g.isLegal(P('f1h'))).toBe(false); // overlaps
    expect(g.isLegal(P('e1v'))).toBe(false); // crosses
    expect(g.isLegal(P('g1h'))).toBe(true);
    g.play(P('e8')); // black move
    expect(moves(g)).toEqual(['d1', 'f1']);
  });

  it('cannot fully block a player', () => {
    const g = new Game();
    // Build a wall across rank 1/2 boundary except one gap, then try to close it
    const seq = ['a1h', 'a9', 'c1h', 'b9', 'e1h', 'a9', 'g1h'];
    for (const s of seq) g.play(P(s));
    // white walls cover a-h; i-file open. Closing it with a vertical on h1 would trap white
    // White is on e1; black to move tries h1v (between h and i, rank 1-2)
    expect(g.isLegal(P('h1v'))).toBe(false);
    expect(g.wallMoves().includes(P('h1v'))).toBe(false);
  });

  it('wallMoves agrees with brute-force isLegal on random games', () => {
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let game = 0; game < 30; game++) {
      const g = new Game();
      while (!g.over && g.ply < 80) {
        const wm = new Set(g.wallMoves());
        for (let a = 81; a < NUM_ACTIONS; a++) expect(wm.has(a)).toBe(g.isLegal(a));
        const legal = g.legalActions();
        g.play(legal[Math.floor(rnd() * legal.length)]);
      }
    }
  });

  it('detects win', () => {
    const g = new Game();
    playAll(g, ['e2', 'd9', 'e3', 'c9', 'e4', 'd9', 'e5', 'c9', 'e6', 'd9', 'e7', 'c9', 'e8', 'd9']);
    expect(g.over).toBe(false);
    g.play(P('e9'));
    expect(g.winner).toBe(0);
  });
});
