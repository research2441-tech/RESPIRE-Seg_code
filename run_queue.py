"""Resumable job queue: runs shell jobs from a list with N single-thread workers.

A job is skipped when its marker file already exists, so the queue can be
restarted at any time without repeating finished work.
"""
import os
import subprocess
import sys
import time
from multiprocessing import Pool

RUNS = os.environ.get("RESPIRE_RUNS", "/home/claude/runs")


def run(job):
    marker, cmd = job
    if marker and os.path.exists(marker):
        return f"skip {cmd}"
    t = time.time()
    env = dict(os.environ, THREADS="1", NPROC="1", OMP_NUM_THREADS="1")
    r = subprocess.run(cmd, shell=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(l for l in r.stdout.splitlines() if "Warning" not in l)[-600:]
    msg = f"[{time.strftime('%H:%M:%S')}] rc={r.returncode} {time.time() - t:.0f}s {cmd}\n{tail}\n"
    with open(os.path.join(RUNS, "queue.log"), "a") as f:
        f.write(msg)
    return msg


def load(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        marker, cmd = line.split("|", 1)
        jobs.append((marker.strip() or None, cmd.strip()))
    return jobs


if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    jobs = load(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with Pool(workers) as p:
        for m in p.imap(run, jobs):
            print(m.splitlines()[0], flush=True)
