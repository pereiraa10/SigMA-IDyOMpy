"""
run_paradigm4_across_subject_across_stimulus.py — across-subject,
across-stimulus generalization holdout (the hardest paradigm).

5 folds, zipping a musician-balanced subject-fold (paradigm3-style, but 4
held-out subjects per fold: 2 musicians + 2 non-musicians) with a
deterministic song-fold. Train on 16 subjects x 7 songs (all repeats), val
on the SAME 16 subjects x 1 held-out song, test on the remaining 4 UNSEEN
subjects x 2 UNSEEN songs -- test subjects AND test songs are both entirely
absent from train. See splits.paradigm4_across_subject_across_stimulus's
docstring for the exact 336/48/24-per-fold arithmetic (408 total, NOT
600 -- intentional, not a bug: see that docstring for why).

Purpose: the hardest test -- requires learning a genuinely generalizable
acoustic-to-neural mapping, not tied to any specific person or song.

Usage:
    python run_paradigm4_across_subject_across_stimulus.py                  # all 5 folds
    python run_paradigm4_across_subject_across_stimulus.py --fold 1         # just fold 1
    python run_paradigm4_across_subject_across_stimulus.py --smoke-test     # fast wiring check
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from dataset_pool import SubjectCache
from manifest import build_manifest
from splits import paradigm4_across_subject_across_stimulus
from run_common import build_arg_parser, run_all_folds

PARADIGM_NAME = 'paradigm4_across_subject_across_stimulus'


def main():
    args = build_arg_parser(__doc__).parse_args(sys.argv[1:])
    config = load_config(cli_args=sys.argv[1:])
    manifest = build_manifest(config)

    folds = paradigm4_across_subject_across_stimulus(config, manifest)
    cache = SubjectCache(config, debug=args.smoke_test)

    run_all_folds(PARADIGM_NAME, folds, config, cache, args, model_family='conv')


if __name__ == '__main__':
    main()
