"""
run_paradigm2_within_subject_across_stimulus.py — within-subject,
across-stimulus generalization holdout.

10 folds, one per held-out song rotation: train on all 20 subjects x 8
songs (all repeats), val on all 20 subjects x 1 held-out song, test on all
20 subjects x 1 different held-out song. See
splits.paradigm2_within_subject_across_stimulus's docstring for the exact
480/60/60-per-fold arithmetic.

Purpose: can the model generalize to novel acoustic content for brains it
has already seen? The standard leave-stimuli-out test.

Usage:
    python run_paradigm2_within_subject_across_stimulus.py                  # all 10 folds
    python run_paradigm2_within_subject_across_stimulus.py --fold 3         # just fold 3
    python run_paradigm2_within_subject_across_stimulus.py --smoke-test     # fast wiring check
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from dataset_pool import SubjectCache
from manifest import build_manifest
from splits import paradigm2_within_subject_across_stimulus
from run_common import build_arg_parser, run_all_folds

PARADIGM_NAME = 'paradigm2_within_subject_across_stimulus'


def main():
    args = build_arg_parser(__doc__).parse_args(sys.argv[1:])
    config = load_config(cli_args=sys.argv[1:])
    manifest = build_manifest(config)

    folds = paradigm2_within_subject_across_stimulus(config, manifest)
    cache = SubjectCache(config, debug=args.smoke_test)

    run_all_folds(PARADIGM_NAME, folds, config, cache, args, model_family='conv')


if __name__ == '__main__':
    main()
