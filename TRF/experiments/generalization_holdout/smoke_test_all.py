"""
smoke_test_all.py — runs every run_paradigmN_*.py script and the null
baseline with --smoke-test, and prints a single PASS/FAIL summary. This is
the one command to run for an end-to-end sanity check (manifest -> fold
split -> dataset build -> training -> eval -> pickle save, all wired
together) before attempting any full-scale (many-hour) real training run.

Usage:
    python smoke_test_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    'run_paradigm1_memorization_ceiling.py',
    'run_paradigm2_within_subject_across_stimulus.py',
    'run_paradigm3_across_subject_within_stimulus.py',
    'run_paradigm4_across_subject_across_stimulus.py',
    'null_baseline_mismatched_stimulus.py',
]

HERE = Path(__file__).resolve().parent


def main():
    results = {}
    for script in SCRIPTS:
        print(f"\n{'=' * 70}\nRunning: python {script} --smoke-test\n{'=' * 70}")
        proc = subprocess.run(
            [sys.executable, str(HERE / script), '--smoke-test'],
            cwd=HERE, capture_output=True, text=True,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:])
        results[script] = proc.returncode == 0

    print(f"\n{'=' * 70}\nSMOKE TEST SUMMARY\n{'=' * 70}")
    all_passed = True
    for script, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"  [{status}] {script}")
        all_passed = all_passed and passed

    if all_passed:
        print("\nALL SMOKE TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME SMOKE TESTS FAILED -- see output above")
        sys.exit(1)


if __name__ == '__main__':
    main()
