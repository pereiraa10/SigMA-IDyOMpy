"""
null_baseline_mismatched_stimulus.py — chance-level ceiling via mismatched
(EEG, stimulus) pairing.

Trains the SAME StimToEEG architecture as the main paradigms, on the SAME
musician-balanced leave-subjects-out splits as paradigm 3
(splits.paradigm3_across_subject_within_stimulus, imported unchanged -- same
5 folds, same 16/2/2 subject split, same GroupKFold-per-stratum
construction), with the SAME train_pooled/evaluate_full_trials harness and
early-stopping criterion -- but every EEG trial is paired with a WRONG
stimulus instead of the one that was actually playing, so a high r here
would indicate the model is fitting something other than genuine
stimulus->response structure (e.g. subject-specific EEG autocorrelation
alone).

Two mismatch strategies, sampled per trial-key (mode='combined' in effect):
    'adjacent'        : same subject, same song, a DIFFERENT repeat, shifted
                         +-0.5..2s in time.
    'cross_stimulus'   : same subject, a DIFFERENT song, any repeat, no shift.

Each trial's mismatched feature array is built ONCE per (fold, feature_set)
up front (NullPairingTable), not resampled every epoch, so the "wrong"
pairing is stable and auditable via log_fold_manifest-style output.

Verification: a hard assert on every constructed pair that the mismatched
(song_id, repeat_idx, shift_samples) tuple never equals the true
(song_id, repeat_idx, 0) alignment (see make_mismatched_pair), plus a soft
warning if the mismatched segment's envelope happens to correlate with the
true trial's envelope beyond a heads-up threshold (musical self-similarity
could in principle cause that even with correct index-level mismatch).

Usage:
    python null_baseline_mismatched_stimulus.py                  # all 5 folds
    python null_baseline_mismatched_stimulus.py --variants nonlinear   # validate one architecture
    python null_baseline_mismatched_stimulus.py --smoke-test     # fast wiring check
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config

from manifest import build_manifest, index_by_subject, index_by_subject_song
from splits import paradigm3_across_subject_within_stimulus
from dataset_pool import SubjectCache
from run_common import build_arg_parser, run_all_folds

PARADIGM_NAME = 'null_baseline'
MISMATCH_STRATEGY_DESC = (
    "combined per-trial sampling of 'adjacent' (same subject, same song, "
    "different repeat, shifted +-0.5..2s) and 'cross_stimulus' (same "
    "subject, different song, any repeat, no shift)"
)
ENVELOPE_WARN_THRESHOLD = 0.3


def _shift_and_match_length(X, shift_samples, target_len):
    """Circular-shift X along time, then crop/edge-pad to exactly target_len
    rows so it can stand in for a trial of that length."""
    X = np.roll(X, shift_samples, axis=0)
    T = X.shape[0]
    if T >= target_len:
        return X[:target_len]
    pad = np.repeat(X[-1:], target_len - T, axis=0)
    return np.concatenate([X, pad], axis=0)


def _warn_if_envelope_correlated(true_ref, source_ref, ds_by_subject, X_wrong, feature_keys,
                                  threshold=ENVELOPE_WARN_THRESHOLD):
    """Soft diagnostic, not a hard failure: known limitation that
    index-level mismatch checks can't rule out spurious envelope
    correlation from musical self-similarity (e.g. a repeated phrase)."""
    if 'envelope' not in feature_keys:
        return
    idx = feature_keys.index('envelope')
    true_env = ds_by_subject[true_ref.subject].trials[true_ref.trial_idx]['envelope']
    wrong_env = X_wrong[:, idx]
    n = min(len(true_env), len(wrong_env))
    if n < 2:
        return
    with np.errstate(invalid='ignore'):
        r = np.corrcoef(true_env[:n], wrong_env[:n])[0, 1]
    if np.isfinite(r) and abs(r) > threshold:
        print(f"  [warn] mismatched pair envelope |r|={abs(r):.3f} > {threshold} for "
              f"{true_ref.subject}#{true_ref.trial_idx} <- "
              f"{source_ref.subject}#{source_ref.trial_idx} "
              f"(musical self-similarity, not a realignment bug)")


def make_mismatched_pair(true_ref, ds_by_subject, by_subject, by_subject_song,
                          feature_keys, sfreq, rng):
    """Build one wrong-stimulus feature array, length-matched to true_ref's
    own trial. Returns (X_wrong, source_ref, meta)."""
    subject = true_ref.subject
    T_true = ds_by_subject[subject].trials[true_ref.trial_idx]['eeg'].shape[0]

    mode = rng.choice(['adjacent', 'cross_stimulus'])
    if mode == 'adjacent':
        candidates = [r for r in by_subject_song[(subject, true_ref.song_id)]
                      if r.repeat_idx != true_ref.repeat_idx]
        source = candidates[rng.integers(len(candidates))]
        shift_sec = float(rng.uniform(0.5, 2.0)) * float(rng.choice([-1.0, 1.0]))
        shift_samples = int(round(shift_sec * sfreq))
        if shift_samples == 0:
            # Can't happen at sfreq=64 (min |shift_sec|=0.5 -> min |shift_samples|=32),
            # but guard the invariant explicitly rather than relying on the arithmetic alone.
            shift_samples = 1
    elif mode == 'cross_stimulus':
        candidates = [r for r in by_subject[subject] if r.song_id != true_ref.song_id]
        source = candidates[rng.integers(len(candidates))]
        shift_samples = 0
    else:
        raise ValueError(mode)

    # Hard verification: the mismatched pair must never accidentally realign
    # with the true stimulus. 'adjacent' forces repeat_idx != true and
    # |shift_samples| >= 1 (guaranteed >=32 at 64Hz by construction above);
    # 'cross_stimulus' forces song_id != true.song_id -- neither can collide
    # with (true.song_id, true.repeat_idx, shift=0).
    assert (source.song_id, source.repeat_idx, shift_samples) != (true_ref.song_id, true_ref.repeat_idx, 0), \
        (f"mismatched pair accidentally identical to true alignment: "
         f"true={true_ref} source={source} mode={mode} shift={shift_samples}")

    src_trial = ds_by_subject[source.subject].trials[source.trial_idx]
    X_src = np.column_stack([src_trial[k] for k in feature_keys])
    X_wrong = _shift_and_match_length(X_src, shift_samples, target_len=T_true)

    _warn_if_envelope_correlated(true_ref, source, ds_by_subject, X_wrong, feature_keys)
    return X_wrong, source, {'mode': mode, 'shift_samples': shift_samples}


class NullPairingTable:
    """{(subject, trial_idx): X_wrong} built once per (fold, feature_set),
    covering every trial key touched by that fold (train+val+test union)."""

    def __init__(self, ds_by_subject, manifest, trial_keys, feature_keys, sfreq, seed=0):
        rng = np.random.default_rng(seed)
        by_subject = index_by_subject(manifest)
        by_subject_song = index_by_subject_song(manifest)
        by_key = {r.key: r for r in manifest}

        self.pairing = {}
        self.pair_meta = {}
        n_asserted = 0
        for key in trial_keys:
            true_ref = by_key[key]
            X_wrong, source, meta = make_mismatched_pair(
                true_ref, ds_by_subject, by_subject, by_subject_song, feature_keys, sfreq, rng)
            self.pairing[key] = X_wrong
            self.pair_meta[key] = {'source_subject': source.subject,
                                    'source_trial_idx': source.trial_idx, **meta}
            n_asserted += 1
        print(f"  [null_baseline] built mismatched pairing for {n_asserted} trials "
              f"(verification assert passed for all)")

    def summary(self):
        """Compact per-mode counts, stored in extra_meta for auditability."""
        modes = [m['mode'] for m in self.pair_meta.values()]
        return {'n_pairs': len(modes),
                'n_adjacent': modes.count('adjacent'),
                'n_cross_stimulus': modes.count('cross_stimulus')}


class MismatchedMultiSubjectWindowDataset(TorchDataset):
    """Same window granularity as dataset_pool.MultiSubjectWindowDataset
    (reuses TRFDataset._index/windows_for_trial, so windows still never
    cross a trial boundary), but slices X from a precomputed mismatched
    feature array instead of the trial's real feature columns. Y is always
    the real EEG for that (subject, trial_idx) -- only the stimulus side is
    wrong."""

    def __init__(self, ds_by_subject, trial_keys, pairing_table):
        self.ds_by_subject = ds_by_subject
        self.pairing_table = pairing_table
        self.window_refs = []   # (subject, trial_idx, start, end)
        for subject, trial_idx in trial_keys:
            ds = ds_by_subject[subject]
            for w in ds.windows_for_trial(trial_idx):
                _, start, end = ds._index[w]
                self.window_refs.append((subject, trial_idx, start, end))

    def __len__(self):
        return len(self.window_refs)

    def __getitem__(self, idx):
        subject, trial_idx, start, end = self.window_refs[idx]
        X_wrong_full = self.pairing_table.pairing[(subject, trial_idx)]
        X = X_wrong_full[start:end].T                                       # (n_feat, win)
        Y = self.ds_by_subject[subject].trials[trial_idx]['eeg'][start:end].T  # (n_ch, win)
        return (torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)))


def main():
    args = build_arg_parser(__doc__).parse_args(sys.argv[1:])
    args.variants = args.variants if '--variants' in sys.argv else 'nonlinear'
    config = load_config(cli_args=sys.argv[1:])
    manifest = build_manifest(config)

    # Same 5 folds as paradigm 3 -- only the feature pairing changes. Retag
    # fold.paradigm (Fold is a mutable dataclass) so fold_dir() writes under
    # results/null_baseline/foldN/ instead of nesting inside paradigm 3's own
    # results directory; assert_fold_integrity already ran (and passed)
    # under the original paradigm name during construction, so this is safe.
    folds = paradigm3_across_subject_within_stimulus(config, manifest)
    for fold in folds:
        fold.paradigm = PARADIGM_NAME
    cache = SubjectCache(config, debug=args.smoke_test)

    seed_holder = {'seed': 0}

    def make_window_dataset_cls(ds_by_subject, feature_keys, fold):
        pairing_table = NullPairingTable(
            ds_by_subject, manifest,
            sorted(set(fold.train_keys) | set(fold.val_keys) | set(fold.test_keys)),
            feature_keys, config.sfreq, seed=seed_holder['seed'] + fold.fold_idx)
        pairing_table_holder[fold.fold_idx] = pairing_table
        return partial(MismatchedMultiSubjectWindowDataset, pairing_table=pairing_table)

    pairing_table_holder = {}

    run_all_folds(
        PARADIGM_NAME, folds, config, cache, args, model_family='conv_null',
        extra_meta_base={'null_baseline': True, 'mismatch_strategy': MISMATCH_STRATEGY_DESC},
        make_window_dataset_cls=make_window_dataset_cls,
    )

    if args.smoke_test:
        for fold_idx, table in pairing_table_holder.items():
            summary = table.summary()
            assert summary['n_pairs'] > 0, f"fold {fold_idx}: no mismatched pairs built"
            print(f"[null_baseline] fold {fold_idx} mismatch summary: {summary}")
        print("[null_baseline] mismatch-verification asserts passed for every sampled pair")


if __name__ == '__main__':
    main()
