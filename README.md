# Quoridor Zero

Browser Quoridor with an AlphaZero-style engine: a policy/value neural network plus Monte-Carlo tree search, trained by self-play. Everything runs client-side. The engine lives in a Web Worker, so the board stays responsive while it thinks.

## Modes

| Mode | What it does |
| --- | --- |
| **vs Computer** | Play the trained engine as White, Black or a random side, at five strengths: Novice, Casual, Club, Expert, Master. |
| **Two Players** | Pass-and-play on one screen. Hints are still available. |
| **Coach** | Play the engine while every one of your moves is graded (Best / Excellent / Good / Inaccuracy / Mistake / Blunder). You see the engine's preferred move and line, plus separate *opening* and *overall* accuracy scores. |
| **Openings → Explore** | Play both sides freely. The engine ranks every candidate with win % and its expected line, served instantly from a precomputed opening book where possible. Click a suggestion to play it. |
| **Openings → Drill** | Opening training. Pick a side; the engine plays varied but sound book replies, and you have to find a good move each turn. A wrong move gets taken back with an explanation so you can retry. Your first-try accuracy is tracked. |

Anywhere you're on move: **Hint** (or `H`) shows the top three moves on the board and in the side panel. **Live analysis** keeps an eval bar and candidate list running. Other keys: `←` undo, `→` redo, `F` flip.

## Running it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # rules engine + network gradient tests
npm run build      # static site in dist/
```

`.github/workflows/pages.yml` deploys `dist/` to GitHub Pages on pushes to `main`. To use it, enable Pages with source "GitHub Actions" in the repo settings.

## How the AI works

- **Rules engine** (`src/engine/quoridor.ts`): bit-cheap board representation and full rules (jumps, diagonal jumps, wall overlap/crossing, path-blocking check). Legal-wall generation only path-checks walls that touch a current shortest path. It's verified against brute force in the tests.
- **Network** (`src/ai/net.ts`): a 345k-parameter MLP with a policy head (209 actions = 81 pawn squares + 128 walls) and a tanh value head. Forward pass, backprop and Adam are hand-written in TypeScript, so training (Node) and play (browser) share one implementation and there are no ML dependencies. Inputs are always from the side-to-move's perspective (Black's view is flipped). They include pawn positions, walls, both players' BFS distance maps and wall counts.
- **Search** (`src/ai/mcts.ts`): PUCT MCTS as in AlphaZero, with first-play urgency, Dirichlet root noise for self-play, tree reuse, and a shortcut for immediate wins.
- **Training** (`training/`): self-play across worker threads. KataGo-style playout-cap randomisation: only full searches become policy targets. Value targets blend the game result with the root Q. There's left/right mirror augmentation and gating, where a new network has to match or beat the previous one before it's promoted.
  - *Bootstrap*: the first iterations use a hand-written path-race heuristic in place of the network. This is a deliberate departure from pure AlphaZero: with one laptop's worth of CPU, starting from random play would take far too long. Its influence on the value is then annealed to zero.

### Training your own model

```bash
npm run train -- --iters 24 --bootstrap 3 --games 128 --sims 200      # writes public/models/model.json
npm run arena -- --a public/models/model.json --b heuristic --games 40 # head-to-head check
npm run book  -- --sims 4000 --depth 9                                  # rebuild public/models/book.json
```

Checkpoints and a log go to `training/runs/` (git-ignored). Resume with `--resume training/runs/genNNN.json`. More iterations, games and sims all make it stronger. The architecture (`DEFAULT_CONFIG` in `net.ts`) can also be grown, at the cost of browser think time.

## Notation

Squares run `a1`–`i9`, with White starting on `e1` and Black on `e9`. A wall is named by the square at its lower-left plus its orientation: `e3h` is a horizontal wall above `e3`/`f3`, and `e3v` is a vertical wall right of `e3`/`e4`.
