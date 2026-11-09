/**
 * A small policy/value multilayer perceptron with hand-written forward/backward passes and Adam.
 * Pure TypeScript so the exact same code runs in Node (training) and the browser (play).
 *
 *   input -> [Linear+ReLU] x hidden.length -> policy head (Linear -> NUM_ACTIONS logits)
 *                                          -> value head  (Linear+ReLU -> Linear -> tanh)
 */
import { NUM_ACTIONS } from '../engine/quoridor.ts';
import { INPUT_SIZE } from './features.ts';

export interface NetConfig {
  input: number;
  hidden: number[];
  valueHidden: number;
  actions: number;
}

export const DEFAULT_CONFIG: NetConfig = {
  input: INPUT_SIZE,
  hidden: [384, 256],
  valueHidden: 64,
  actions: NUM_ACTIONS,
};

class Linear {
  W: Float32Array; // [in][out]
  b: Float32Array;
  gW: Float32Array;
  gb: Float32Array;
  readonly nIn: number;
  readonly nOut: number;
  constructor(nIn: number, nOut: number, params: Float32Array, offset: number) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.W = params.subarray(offset, offset + nIn * nOut);
    this.b = params.subarray(offset + nIn * nOut, offset + nIn * nOut + nOut);
    this.gW = new Float32Array(0);
    this.gb = new Float32Array(0);
  }
  static size(nIn: number, nOut: number) {
    return nIn * nOut + nOut;
  }

  /** Y[B][out] = X[B][in] * W + b */
  forward(X: Float32Array, Y: Float32Array, B: number) {
    const { nIn, nOut, W, b } = this;
    for (let s = 0; s < B; s++) {
      const yo = s * nOut;
      for (let j = 0; j < nOut; j++) Y[yo + j] = b[j];
      const xo = s * nIn;
      for (let i = 0; i < nIn; i++) {
        const x = X[xo + i];
        if (x === 0) continue;
        const wo = i * nOut;
        for (let j = 0; j < nOut; j++) Y[yo + j] += x * W[wo + j];
      }
    }
  }

  /** Accumulates gradients; writes dX if given. */
  backward(X: Float32Array, dY: Float32Array, dX: Float32Array | null, B: number) {
    const { nIn, nOut, W, gW, gb } = this;
    for (let s = 0; s < B; s++) {
      const yo = s * nOut;
      for (let j = 0; j < nOut; j++) gb[j] += dY[yo + j];
      const xo = s * nIn;
      for (let i = 0; i < nIn; i++) {
        const x = X[xo + i];
        const wo = i * nOut;
        if (x !== 0) for (let j = 0; j < nOut; j++) gW[wo + j] += x * dY[yo + j];
        if (dX) {
          let acc = 0;
          for (let j = 0; j < nOut; j++) acc += dY[yo + j] * W[wo + j];
          dX[xo + i] = acc;
        }
      }
    }
  }
}

export interface NetOutput {
  /** raw policy logits in canonical action space */
  logits: Float32Array;
  /** value in [-1, 1] from the perspective of the side to move */
  value: number;
}

export interface TrainBatch {
  x: Float32Array; // [B][input]
  pi: Float32Array; // [B][actions] target distribution (zeros on illegal)
  mask: Uint8Array; // [B][actions] legal mask
  z: Float32Array; // [B] value target
  B: number;
}

export class Net {
  readonly config: NetConfig;
  params: Float32Array;
  private trunk: Linear[] = [];
  private pHead: Linear;
  private vHidden: Linear;
  private vOut: Linear;
  // Adam state
  private m: Float32Array | null = null;
  private v: Float32Array | null = null;
  private grads: Float32Array | null = null;
  private t = 0;
  private bufCap = 0;
  private acts: Float32Array[] = [];

  constructor(config: NetConfig = DEFAULT_CONFIG, params?: Float32Array) {
    this.config = config;
    const total = Net.paramCount(config);
    this.params = params ?? new Float32Array(total);
    if (this.params.length !== total) throw new Error(`param size mismatch ${this.params.length} != ${total}`);
    let off = 0;
    let prev = config.input;
    for (const h of config.hidden) {
      this.trunk.push(new Linear(prev, h, this.params, off));
      off += Linear.size(prev, h);
      prev = h;
    }
    this.pHead = new Linear(prev, config.actions, this.params, off);
    off += Linear.size(prev, config.actions);
    this.vHidden = new Linear(prev, config.valueHidden, this.params, off);
    off += Linear.size(prev, config.valueHidden);
    this.vOut = new Linear(config.valueHidden, 1, this.params, off);
    if (!params) this.initWeights();
  }

  static paramCount(c: NetConfig): number {
    let n = 0;
    let prev = c.input;
    for (const h of c.hidden) {
      n += Linear.size(prev, h);
      prev = h;
    }
    return n + Linear.size(prev, c.actions) + Linear.size(prev, c.valueHidden) + Linear.size(c.valueHidden, 1);
  }

  private layers(): Linear[] {
    return [...this.trunk, this.pHead, this.vHidden, this.vOut];
  }

  private initWeights(seed = 12345) {
    let s = seed;
    const rnd = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    for (const l of this.layers()) {
      const std = Math.sqrt(2 / l.nIn);
      const scale = l === this.pHead || l === this.vOut ? 0.1 : 1;
      for (let i = 0; i < l.W.length; i++) l.W[i] = gauss() * std * scale;
      l.b.fill(0);
    }
  }

  private ensureBuffers(B: number) {
    if (B <= this.bufCap) return;
    this.bufCap = B;
    const c = this.config;
    this.acts = [];
    // acts[0..H-1] = trunk outputs (post-ReLU), then policy logits, value hidden, value pre-tanh
    for (const h of c.hidden) this.acts.push(new Float32Array(B * h));
    this.acts.push(new Float32Array(B * c.actions));
    this.acts.push(new Float32Array(B * c.valueHidden));
    this.acts.push(new Float32Array(B));
  }

  private forwardBatch(x: Float32Array, B: number) {
    this.ensureBuffers(B);
    let inp = x;
    for (let i = 0; i < this.trunk.length; i++) {
      const out = this.acts[i];
      this.trunk[i].forward(inp, out, B);
      for (let k = 0, n = B * this.trunk[i].nOut; k < n; k++) if (out[k] < 0) out[k] = 0;
      inp = out;
    }
    const H = this.trunk.length;
    this.pHead.forward(inp, this.acts[H], B);
    const vh = this.acts[H + 1];
    this.vHidden.forward(inp, vh, B);
    for (let k = 0, n = B * this.config.valueHidden; k < n; k++) if (vh[k] < 0) vh[k] = 0;
    this.vOut.forward(vh, this.acts[H + 2], B);
  }

  /** Single-position inference. Returned arrays are reused between calls — copy if you keep them. */
  predict(x: Float32Array): NetOutput {
    this.forwardBatch(x, 1);
    const H = this.trunk.length;
    return { logits: this.acts[H].subarray(0, this.config.actions), value: Math.tanh(this.acts[H + 2][0]) };
  }

  /**
   * One Adam step on a batch. Returns { policyLoss, valueLoss }.
   */
  trainStep(batch: TrainBatch, lr: number, weightDecay = 1e-4, valueWeight = 1): { policyLoss: number; valueLoss: number } {
    const { x, pi, mask, z, B } = batch;
    const c = this.config;
    const A = c.actions;
    this.forwardBatch(x, B);
    const H = this.trunk.length;
    const logits = this.acts[H];
    const vh = this.acts[H + 1];
    const vpre = this.acts[H + 2];

    if (!this.grads) {
      this.grads = new Float32Array(this.params.length);
      this.m = new Float32Array(this.params.length);
      this.v = new Float32Array(this.params.length);
      // bind gradient views
      let off = 0;
      for (const l of this.layers()) {
        const wn = l.nIn * l.nOut;
        l.gW = this.grads.subarray(off, off + wn);
        l.gb = this.grads.subarray(off + wn, off + wn + l.nOut);
        off += wn + l.nOut;
      }
    }
    this.grads.fill(0);

    // ---- policy loss: masked softmax cross-entropy
    const dLogits = new Float32Array(B * A);
    let pLoss = 0;
    for (let s = 0; s < B; s++) {
      const o = s * A;
      let mx = -Infinity;
      for (let a = 0; a < A; a++) if (mask[o + a] && logits[o + a] > mx) mx = logits[o + a];
      let sum = 0;
      for (let a = 0; a < A; a++) if (mask[o + a]) sum += Math.exp(logits[o + a] - mx);
      const lse = mx + Math.log(sum);
      for (let a = 0; a < A; a++) {
        if (!mask[o + a]) continue;
        const logp = logits[o + a] - lse;
        const t = pi[o + a];
        if (t > 0) pLoss -= t * logp;
        dLogits[o + a] = (Math.exp(logp) - t) / B;
      }
    }
    // ---- value loss: MSE on tanh output
    const dV = new Float32Array(B);
    let vLoss = 0;
    for (let s = 0; s < B; s++) {
      const v = Math.tanh(vpre[s]);
      const d = v - z[s];
      vLoss += d * d;
      dV[s] = ((2 * d * (1 - v * v)) / B) * valueWeight;
    }

    // ---- backward
    const trunkOut = H > 0 ? this.acts[H - 1] : x;
    const trunkW = H > 0 ? this.trunk[H - 1].nOut : c.input;
    const dTrunk = new Float32Array(B * trunkW);
    const dVh = new Float32Array(B * c.valueHidden);
    this.vOut.backward(vh, dV, dVh, B);
    for (let k = 0; k < dVh.length; k++) if (vh[k] <= 0) dVh[k] = 0;
    const tmp = new Float32Array(B * trunkW);
    this.vHidden.backward(trunkOut, dVh, tmp, B);
    this.pHead.backward(trunkOut, dLogits, dTrunk, B);
    for (let k = 0; k < dTrunk.length; k++) dTrunk[k] += tmp[k];

    let dOut = dTrunk;
    for (let i = H - 1; i >= 0; i--) {
      const out = this.acts[i];
      for (let k = 0; k < dOut.length; k++) if (out[k] <= 0) dOut[k] = 0;
      const inp = i > 0 ? this.acts[i - 1] : x;
      const dIn = i > 0 ? new Float32Array(B * this.trunk[i - 1].nOut) : null;
      this.trunk[i].backward(inp, dOut, dIn, B);
      if (dIn) dOut = dIn;
    }

    // ---- Adam
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t);
    const c2 = 1 - Math.pow(b2, this.t);
    const p = this.params;
    const g = this.grads;
    const m = this.m!;
    const v = this.v!;
    for (let i = 0; i < p.length; i++) {
      const gi = g[i] + weightDecay * p[i];
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
    }
    return { policyLoss: pLoss / B, valueLoss: vLoss / B };
  }

  // ---------- serialization ----------
  toJSON(meta: Record<string, unknown> = {}): string {
    const bytes = new Uint8Array(this.params.buffer, this.params.byteOffset, this.params.byteLength);
    return JSON.stringify({ format: 'quoridor-zero-mlp-v1', config: this.config, meta, params: bytesToBase64(bytes) });
  }

  static fromJSON(json: string | { config: NetConfig; params: string }): Net {
    const obj = typeof json === 'string' ? JSON.parse(json) : json;
    const bytes = base64ToBytes(obj.params);
    const params = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
    return new Net(obj.config, params);
  }
}

function bytesToBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function base64ToBytes(b64: string): Uint8Array {
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(b64, 'base64');
    return new Uint8Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  }
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}
