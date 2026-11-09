import { Game } from '../src/engine/quoridor.ts';
import { actionToString } from '../src/engine/notation.ts';
import { HeuristicEvaluator, NetEvaluator } from '../src/ai/evaluator.ts';
import { MCTS } from '../src/ai/mcts.ts';
import { Net } from '../src/ai/net.ts';

const g = new Game();
for (const [name, ev] of [['heuristic', new HeuristicEvaluator()], ['net', new NetEvaluator(new Net())]] as const) {
  const t = performance.now();
  const m = new MCTS(g, ev);
  m.search(2000);
  const dt = performance.now() - t;
  const r = m.result(5);
  console.log(name, `${(2000 / dt * 1000).toFixed(0)} sims/s`, r.value.toFixed(3), r.moves.map(x => `${actionToString(x.action)}:${x.visits}:${x.q.toFixed(2)}`).join(' '));
}
// heuristic self-play game
const t = performance.now();
const game = new Game();
const ev = new HeuristicEvaluator();
const seq: string[] = [];
while (!game.over) {
  const m = new MCTS(game, ev);
  m.search(300);
  const a = m.pick(0);
  seq.push(actionToString(a));
  game.play(a);
}
console.log('game', game.winner, game.ply, ((performance.now() - t) / 1000).toFixed(1) + 's', seq.join(' '));
