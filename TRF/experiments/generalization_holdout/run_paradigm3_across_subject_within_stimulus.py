"""
run_paradigm3_across_subject_within_stimulus.py — across-subject,
within-stimulus generalization holdout.

5 folds via sklearn.model_selection.GroupKFold(groups=subject_id) run
separately within each subject_type stratum (musician / non-musician) and
zipped, guaranteeing every fold's val_subjects and test_subjects are each
[1 musician, 1 non-musician]. Train on 16 subjects x all 10 songs (all
repeats), val on 2 subjects (1 musician + 1 non-musician) x all 10 songs,
test on 2 different subjects (1 musician + 1 non-musician) x all 10 songs.
See splits.paradigm3_across_subject_within_stimulus's docstring for the
exact 480/60/60-per-fold arithmetic and the musician-balance construction.

Purpose: can the model adapt to new brains listening to already-seen
content? The standard leave-subjects-out test.

Usage:
    python run_paradigm3_across_subject_within_stimulus.py                  # all 5 folds
    python run_paradigm3_across_subject_within_stimulus.py --fold 2         # just fold 2
    python run_paradigm3_across_subject_within_stimulus.py --smoke-test     # fast wiring check
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from dataset_pool import SubjectCache
from manifest import build_manifest
from splits import paradigm3_across_subject_within_stimulus
from run_common import build_arg_parser, run_all_folds

PARADIGM_NAME = 'paradigm3_across_subject_within_stimulus'


def main():
    args = build_arg_parser(__doc__).parse_args(sys.argv[1:])
    config = load_config(cli_args=sys.argv[1:])
    manifest = build_manifest(config)

    folds = paradigm3_across_subject_within_stimulus(config, manifest)
    cache = SubjectCache(config, debug=args.smoke_test)

    run_all_folds(PARADIGM_NAME, folds, config, cache, args, model_family='conv')


if __name__ == '__main__':
    main()
