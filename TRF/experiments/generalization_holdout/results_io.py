"""
results_io.py — pickle-saving for the generalization_holdout experiments,
reusing results.py's schema/build_result/result_filename/plotting functions
unmodified, but NEVER calling results.save_result() directly.

Why not results.save_result(): it calls plot_alignment(..., raw_units=True)
(default) and plot_topoarray(..., raw_units=True) (default), both of which
invert z-scoring via _reload_raw_eeg_trials(subject) -- which reloads a
subject's FULL 30-trial session and asserts
len(raw_trials) == len(trial_boundaries). That holds for the existing
single-subject full-session scripts, but is false for most of our folds,
where a pickle's trial_boundaries covers only a strict subset of a subject's
30 trials (e.g. paradigm 2's test set is 1 song x 3 repeats = 3 trials).
save_holdout_result below calls plot_alignment/plot_topobutterfly/
plot_topoarray individually with raw_units=False forced on all three,
sidestepping _reload_raw_eeg_trials entirely (pickling logic identical to
results.save_result otherwise).
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, not a package

import results as res

BASE_RESULTS_DIR = Path(__file__).resolve().parent / 'results'


def fold_dir(fold, base_results_dir=BASE_RESULTS_DIR):
    if fold.paradigm == 'paradigm1_memorization_ceiling':
        return base_results_dir / fold.paradigm
    return base_results_dir / fold.paradigm / f'fold{fold.fold_idx}'


def save_holdout_result(path, result, channel_idx=res.DEFAULT_CHANNEL_IDX,
                         sfreq=res.DEFAULT_SFREQ, plot=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    if plot:
        res.plot_alignment(result, path.parent, channel_idx=channel_idx, sfreq=sfreq, raw_units=False)
        res.plot_topobutterfly(result, path.parent, sfreq=sfreq, raw_units=False)
        res.plot_topoarray(result, path.parent, sfreq=sfreq, raw_units=False)
    return path
