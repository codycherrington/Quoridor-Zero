import { describe, expect, it } from 'vitest';
import { Net } from '../src/ai/net.ts';
import { encode, flipAction, mirrorAction, mirrorInput, MIRROR_PERM } from '../src/ai/features.ts';
import { Game, NUM_ACTIONS } from '../src/engine/quoridor.ts';
import { parseAction } from '../src/engine/notation.ts';

describe('features', () => {
  it('flip/mirror are involutions', () => {
    for (let a = 0; a < NUM_ACTIONS; a++) {
      expect(flipAction(flipAction(a))).toBe(a);
      expect(mirrorAction(mirrorAction(a))).toBe(a);
    }
  });
  it('mirrorInput equals encoding of mirrored position', () => {
    const g = new Game();
    for (const s of ['e2', 'e8', 'c3h', 'f6v', 'd2']) g.play(parseAction(s));
    const m = new Game();
    for (const s of ['e2', 'e8', 'c3h', 'f6v', 'd2']) m.play(mirrorAction(parseAction(s)));
    expect(Array.from(mirrorInput(encode(g)))).toEqual(Array.from(encode(m)));
    expect(MIRROR_PERM.length).toBe(NUM_ACTIONS);
  });
});

describe('net', () => {
  it('gradient matches finite differences', () => {
    const cfg = { input: 6, hidden: [5, 4], valueHidden: 3, actions: 4 };
    const net = new Net(cfg);
    const B = 3;
    const x = Float32Array.from({ length: B * 6 }, (_, i) => Math.sin(i * 1.7));
    const pi = new Float32Array([0.5, 0.5, 0, 0, 0, 1, 0, 0, 0.2, 0.3, 0.5, 0]);
    const mask = new Uint8Array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1]);
    const z = new Float32Array([0.5, -0.3, 0.9]);
    const batch = { x, pi, mask, z, B };
    const loss = (n: Net) => {
      const c = new Net(cfg, n.params.slice());
      const r = c.trainStep(batch, 0, 0);
      return r.policyLoss + r.valueLoss;
    };
    // analytic gradient via one SGD-ish probe: use Adam lr=0 trick -> access grads
    const probe = new Net(cfg, net.params.slice());
    probe.trainStep(batch, 0, 0);
    const grads: Float32Array = (probe as any).grads;
    const eps = 1e-3;
    for (const i of [0, 7, 20, 33, 50, net.params.length - 1, net.params.length - 5]) {
      const p1 = new Net(cfg, net.params.slice());
      p1.params[i] += eps;
      const p2 = new Net(cfg, net.params.slice());
      p2.params[i] -= eps;
      const num = (loss(p1) - loss(p2)) / (2 * eps);
      expect(Math.abs(num - grads[i])).toBeLessThan(2e-3 + 0.02 * Math.abs(num));
    }
  });
  it('serializes', () => {
    const net = new Net();
    const back = Net.fromJSON(net.toJSON({ a: 1 }));
    expect(back.params).toEqual(net.params);
  });
});
