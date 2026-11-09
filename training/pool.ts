import { Worker } from 'node:worker_threads';
import os from 'node:os';
import type { Job } from './worker.ts';

const WORKER_URL = new URL('./worker.ts', import.meta.url);

/** Run jobs across worker threads; `onMessage` receives every non-"done" message. */
export async function runJobs(jobs: Job[], onMessage: (msg: any) => void, threads = os.cpus().length): Promise<void> {
  const queue = jobs.slice();
  const n = Math.min(threads, queue.length);
  await Promise.all(
    Array.from({ length: n }, async () => {
      const w = new Worker(WORKER_URL);
      try {
        for (let job = queue.shift(); job; job = queue.shift()) {
          await new Promise<void>((resolve, reject) => {
            const handler = (msg: any) => {
              if (msg.type === 'done') {
                w.off('message', handler);
                w.off('error', reject);
                resolve();
              } else onMessage(msg);
            };
            w.on('message', handler);
            w.once('error', reject);
            w.postMessage(job);
          });
        }
      } finally {
        await w.terminate();
      }
    }),
  );
}
