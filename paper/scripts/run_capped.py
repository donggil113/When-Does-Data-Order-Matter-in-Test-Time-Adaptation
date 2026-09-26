"""Run one command under a CPU-time and address-space cap and append its actual usage to a TSV log.

    python3 paper/scripts/run_capped.py <budget_name> <cpu_seconds> -- <command...>

The child gets RLIMIT_CPU = cpu_seconds and RLIMIT_AS = 3 GiB and one-thread environment variables. Usage
(child user+sys CPU, peak RSS, wall, exit code) is appended to paper/generated/budget_log.tsv so that the
per-budget totals (analysis 600 s; build/static checks 600 s) can be summed.
"""

import os
import resource
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG = os.path.join(ROOT, "paper", "generated", "budget_log.tsv")


def main() -> int:
    name, cpu = sys.argv[1], int(sys.argv[2])
    cmd = sys.argv[sys.argv.index("--") + 1:]

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, preexec_fn=limits, env=env)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    used = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if new:
            f.write("utc\tbudget\tcpu_cap_s\tcpu_used_s\tpeak_rss_mib\twall_s\texit\tcommand\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{name}\t{cpu}\t{used:.2f}\t"
                f"{after.ru_maxrss / 1024:.1f}\t{wall:.2f}\t{p.returncode}\t{' '.join(cmd)}\n")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
