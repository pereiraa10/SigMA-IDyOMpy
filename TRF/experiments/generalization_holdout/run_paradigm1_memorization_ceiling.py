"""
run_paradigm1_memorization_ceiling.py — within-subject, within-stimulus
holdout (memorization / overfitting ceiling).

Single split (not a rotating k-fold): for each subject, most songs
contribute all 3 repeats to training; a stratified 45% of (subject, song)
pairs instead split their 3 repeats one each to train/val/test, keeping
each whole repeat intact. See splits.paradigm1_memorization_ceiling's
docstring for the exact 420/90/90 (70/15/15%) arithmetic.

Purpose: mostly tests memorization capacity / subject-specific fitting, not
genuine generalization -- this is the ceiling the other 3 paradigms are
compared against.

Usage:
    python run_paradigm1_memorization_ceiling.py                 # full run, both variants, all feature_sets
    python run_paradigm1_memorization_ceiling.py --smoke-test     # fast wiring check
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from dataset_pool import SubjectCache
from manifest import build_manifest
from splits import paradigm1_memorization_ceiling
from run_common import build_arg_parser, run_all_folds

PARADIGM_NAME = 'paradigm1_memorization_ceiling'


def main():
    args = build_arg_parser(__doc__).parse_args(sys.argv[1:])
    config = load_config(cli_args=sys.argv[1:])
    manifest = build_manifest(config)

    fold = paradigm1_memorization_ceiling(config, manifest)
    cache = SubjectCache(config, debug=args.smoke_test)

    run_all_folds(PARADIGM_NAME, [fold], config, cache, args, model_family='conv')


if __name__ == '__main__':
    main()
