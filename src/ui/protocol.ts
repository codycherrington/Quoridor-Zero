import type { SearchResult } from '../ai/mcts.ts';

export interface BookEntry {
  value: number;
  visits: number;
  moves: { action: number; visits: number; q: number; pv: number[] }[];
}

export type WorkerRequest =
  | { type: 'move'; id: number; moves: number[]; level: number }
  | { type: 'analyze'; id: number; moves: number[]; sims: number; useBook?: boolean }
  | { type: 'grade'; id: number; moves: number[]; action: number; sims: number }
  | { type: 'cancel' }
  | { type: 'init'; base: string };

export type WorkerResponse =
  | { type: 'ready'; modelInfo: string; bookSize: number }
  | { type: 'move'; id: number; action: number; result: SearchResult }
  | { type: 'analysis'; id: number; result: SearchResult; done: boolean; source: 'book' | 'search' }
  | {
      type: 'grade';
      id: number;
      action: number;
      bestAction: number;
      bestQ: number;
      playedQ: number;
      bestPv: number[];
      result: SearchResult;
    };
