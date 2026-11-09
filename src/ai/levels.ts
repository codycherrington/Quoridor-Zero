export interface Level {
  id: number;
  name: string;
  blurb: string;
  /** MCTS simulations per move (1 = raw network policy) */
  sims: number;
  /** move-selection temperature (0 = always best) */
  temperature: number;
  /** temperature applies only to the first N plies (Infinity = whole game) */
  tempPlies: number;
}

export const LEVELS: Level[] = [
  { id: 1, name: 'Novice', blurb: 'Plays on instinct alone — raw network, no lookahead, lots of variety.', sims: 1, temperature: 1.0, tempPlies: Infinity },
  { id: 2, name: 'Casual', blurb: 'A short look ahead; still makes human-ish slips.', sims: 24, temperature: 0.6, tempPlies: Infinity },
  { id: 3, name: 'Club', blurb: 'Solid search. Punishes obvious mistakes.', sims: 160, temperature: 0.25, tempPlies: 8 },
  { id: 4, name: 'Expert', blurb: 'Deep search, plays its best move every time.', sims: 800, temperature: 0, tempPlies: 0 },
  { id: 5, name: 'Master', blurb: 'Thinks hard (a few seconds per move). Good luck.', sims: 3000, temperature: 0, tempPlies: 0 },
];

export const levelById = (id: number) => LEVELS.find((l) => l.id === id) ?? LEVELS[2];
