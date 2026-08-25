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

Mismatch strategy: cross-song substitution ONLY -- same subject, a
DIFFERENT song, any repeat, with a random circular shift. A same-song
"adjacent repeat, shifted 0.5-2s" mode used to also be sampled 50/50, but
music is periodic at exactly that timescale (beats, bars, repeated
figures), so a short circular shift of a repeat of the SAME song doesn't
meaningfully decorrelate it -- that mode was inflating the null and has
been removed entirely. The random shift applied to the substituted song is
still needed even for cross-song pairs: without it, t=0 of the wrong song
would always align with t=0 of the true trial, and if trials share a fixed
lead-in/cue at onset, that coincidental alignment could itself inflate the
null.

Each trial's mismatched feature array used for TRAINING is built ONCE per
(fold, feature_set) up front (NullPairingTable), not resampled every epoch,
so the "wrong" pairing driving training is stable and auditable via
log_fold_manifest-style output. At EVALUATION time, instead of a single
point-estimate null r, the already-trained model is re-evaluated (no
retraining) against N independent cross-song substitutions per test trial,
building an empirical null distribution of r per subject/channel; the
model's r on its actual test stimulus is then reported as a z-score/
percentile against that distribution (see make_compute_null_stats).

Verification: a hard assert on every constructed pair that the mismatched
(song_id, repeat_idx, shift_samples) tuple never equals the true
(song_id, repeat_idx, 0) alignment (see make_mismatched_pair). Beyond that,
EVERY feature channel (not just envelope) is checked for true-vs-wrong
correlation; if any feature's |r| exceeds FEATURE_CORR_THRESHOLD, the
candidate is redrawn (a different cross-song substitution is tried) rather
than silently accepted, since musical self-similarity (e.g. a repeated
phrase) could in principle produce a spuriously-correlated "wrong" pair
even though the index-level mismatch is correct.

Usage:
    python null_baseline_mismatched_stimulus.py                  # all 5 folds
    python null_baseline_mismatched_stimulus.py --variants nonlinear   # validate one architecture
    python null_baseline_mismatched_stimulus.py --n-null-draws 50      # more null draws
    python null_baseline_mismatched_stimulus.py --smoke-test     # fast wiring check
"""

import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from TRF_conv import _to_tensor

from manifest import build_manifest, index_by_subject
from splits import paradigm3_across_subject_within_stimulus
from dataset_pool import SubjectCache
from run_common import build_arg_parser, run_all_folds

PARADIGM_NAME = 'null_baseline'
MISMATCH_STRATEGY_DESC = (
    "cross-song substitution only (same subject, a DIFFERENT song, any "
    "repeat, with a random circular shift so a shared onset/lead-in can't "
    "coincidentally realign); every feature channel is checked for "
    "true-vs-wrong correlation and redrawn on violation; evaluated against "
    "an N-draw empirical null r distribution per subject/channel, not a "
    "single point estimate"
)
FEATURE_CORR_THRESHOLD = 0.3
MAX_MISMATCH_RETRIES = 10
DEFAULT_N_NULL_DRAWS = 30
SMOKE_TEST_N_NULL_DRAWS = 3


def _shift_and_match_length(X, shift_samples, target_len):
    """Circular-shift X along time, then crop/edge-pad to exactly target_len
    rows so it can stand in for a trial of that length."""
    X = np.roll(X, shift_samples, axis=0)
    T = X.shape[0]
    if T >= target_len:
        return X[:target_len]
    pad = np.repeat(X[-1:], target_len - T, axis=0)
    return np.concatenate([X, pad], axis=0)


def _correlated_features(true_ref, ds_by_subject, X_wrong, feature_keys,
                          threshold=FEATURE_CORR_THRESHOLD):
    """Every feature channel (not just envelope) checked for true-vs-wrong
    correlation. Returns [(feature_name, r), ...] for channels whose |r|
    exceeds `threshold` -- musical self-similarity (e.g. a repeated phrase)
    could in principle produce a spuriously-correlated "wrong" pair even
    though the index-level mismatch (song_id) is correct."""
    true_trial = ds_by_subject[true_ref.subject].trials[true_ref.trial_idx]
    bad = []
    for idx, feat in enumerate(feature_keys):
        true_feat = true_trial[feat]
        wrong_feat = X_wrong[:, idx]
        n = min(len(true_feat), len(wrong_feat))
        if n < 2:
            continue
        with np.errstate(invalid='ignore'):
            r = np.corrcoef(true_feat[:n], wrong_feat[:n])[0, 1]
        if np.isfinite(r) and abs(r) > threshold:
            bad.append((feat, float(r)))
    return bad


def make_mismatched_pair(true_ref, ds_by_subject, by_subject, feature_keys, sfreq, rng,
                          max_retries=MAX_MISMATCH_RETRIES, threshold=FEATURE_CORR_THRESHOLD):
    """Build one cross-song wrong-stimulus feature array, length-matched to
    true_ref's own trial, with a random circular shift. If any feature
    channel's true-vs-wrong correlation exceeds `threshold`, redraws a
    different cross-song candidate (up to `max_retries` times) instead of
    silently accepting the pair. Returns (X_wrong, source_ref, meta), where
    meta['n_redraws'] records how many redraws were needed for this trial
    (a useful diagnostic -- see NullPairingTable.summary)."""
    subject = true_ref.subject
    T_true = ds_by_subject[subject].trials[true_ref.trial_idx]['eeg'].shape[0]
    candidates = [r for r in by_subject[subject] if r.song_id != true_ref.song_id]

    n_redraws = 0
    X_wrong = source = shift_samples = None
    for attempt in range(max_retries + 1):
        source = candidates[rng.integers(len(candidates))]
        src_trial = ds_by_subject[source.subject].trials[source.trial_idx]
        T_src = src_trial['eeg'].shape[0]
        shift_samples = int(rng.integers(0, T_src)) if T_src > 1 else 0

        # Hard verification: a cross-song substitution can never accidentally
        # realign with the true stimulus, since song_id always differs --
        # kept as an explicit guard on the invariant rather than relying on
        # the candidate-filtering above alone.
        assert (source.song_id, source.repeat_idx, shift_samples) != \
               (true_ref.song_id, true_ref.repeat_idx, 0), \
            (f"mismatched pair accidentally identical to true alignment: "
             f"true={true_ref} source={source} shift={shift_samples}")

        X_src = np.column_stack([src_trial[k] for k in feature_keys])
        X_wrong = _shift_and_match_length(X_src, shift_samples, target_len=T_true)

        bad = _correlated_features(true_ref, ds_by_subject, X_wrong, feature_keys, threshold)
        if not bad:
            break
        n_redraws += 1
        if attempt == max_retries:
            print(f"  [warn] {true_ref.subject}#{true_ref.trial_idx}: no clean cross-song "
                  f"substitution found after {max_retries} redraws, using last candidate "
                  f"anyway (correlated features: {[f for f, _ in bad]})")

    return X_wrong, source, {'shift_samples': shift_samples, 'n_redraws': n_redraws}


class NullPairingTable:
    """{(subject, trial_idx): X_wrong} built once per (fold, feature_set),
    covering every trial key touched by that fold (train+val+test union).
    Drives TRAINING only -- evaluation instead draws N independent
    substitutions per test trial at eval time (see make_compute_null_stats),
    since resampling every trial on every epoch would be wasteful and
    doesn't matter for training dynamics the way it does for estimating a
    null distribution."""

    def __init__(self, ds_by_subject, manifest, trial_keys, feature_keys, sfreq, seed=0):
        rng = np.random.default_rng(seed)
        by_subject = index_by_subject(manifest)
        by_key = {r.key: r for r in manifest}

        self.pairing = {}
        self.pair_meta = {}
        n_asserted = 0
        n_redraws_total = 0
        for key in trial_keys:
            true_ref = by_key[key]
            X_wrong, source, meta = make_mismatched_pair(
                true_ref, ds_by_subject, by_subject, feature_keys, sfreq, rng)
            self.pairing[key] = X_wrong
            self.pair_meta[key] = {'source_subject': source.subject,
                                    'source_trial_idx': source.trial_idx, **meta}
            n_asserted += 1
            n_redraws_total += meta['n_redraws']
        print(f"  [null_baseline] built mismatched pairing for {n_asserted} trials "
              f"(verification assert passed for all, {n_redraws_total} redraws "
              f"needed for correlated candidates)")

    def summary(self):
        """Compact counts, stored in extra_meta for auditability."""
        n_redraws = [m['n_redraws'] for m in self.pair_meta.values()]
        return {'n_pairs': len(n_redraws),
                'n_redraws_total': sum(n_redraws),
                'max_redraws_for_one_trial': max(n_redraws) if n_redraws else 0}


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


def _r_per_channel(Y_true, Y_pred):
    return np.array([pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(Y_true.shape[1])])


def make_compute_null_stats(manifest, sfreq, n_draws, base_seed):
    """Returns a compute_null_stats(...) closure suitable for
    run_common.run_fold's `compute_null_stats` hook: re-evaluates the
    ALREADY-TRAINED model (no retraining) against `n_draws` independent
    cross-song substitutions per test trial, building an empirical null r
    distribution per subject/channel. The model's r on its actual test
    stimulus (Y_true vs Y_pred, as already computed by evaluate_full_trials)
    is reported as a z-score/percentile against that null distribution.

    Efficient by construction: training happens once per (fold, feature_set,
    variant) as before; only forward passes (no backward/optimizer step) are
    added here, one per (test_trial, draw).
    """
    by_subject = index_by_subject(manifest)
    by_key = {r.key: r for r in manifest}
    redraw_totals = {'total': 0, 'pairs': 0}

    def compute_null_stats(model, ds_by_subject, feature_keys, test_subject, subj_test_keys,
                            Y_true, Y_pred, trial_boundaries, channel_names, fold, feature_set, variant):
        seed = (base_seed + fold.fold_idx * 1_000_003
                + hash((feature_set, variant, test_subject)) % 1_000_000)
        rng = np.random.default_rng(seed % (2 ** 32))

        real_r = _r_per_channel(Y_true, Y_pred)
        n_channels = Y_true.shape[1]

        model.eval()
        draws = np.zeros((n_draws, n_channels))
        for d in range(n_draws):
            pred_parts = []
            for key in subj_test_keys:
                true_ref = by_key[key]
                X_wrong, _source, meta = make_mismatched_pair(
                    true_ref, ds_by_subject, by_subject, feature_keys, sfreq, rng)
                redraw_totals['total'] += meta['n_redraws']
                redraw_totals['pairs'] += 1
                with torch.no_grad():
                    pred = model(_to_tensor(X_wrong)).cpu().numpy()[0].T   # (T, n_ch)
                pred_parts.append(pred)
            Y_pred_null = np.concatenate(pred_parts)
            draws[d] = _r_per_channel(Y_true, Y_pred_null)

        null_mean = draws.mean(axis=0)
        null_std = draws.std(axis=0, ddof=1) if n_draws > 1 else np.zeros(n_channels)
        with np.errstate(invalid='ignore', divide='ignore'):
            z = np.where(null_std > 1e-12, (real_r - null_mean) / null_std, np.nan)
        percentile = (draws <= real_r[None, :]).mean(axis=0) * 100.0

        return {
            'null_r_all_draws': draws,
            'null_r_distribution_mean': null_mean,
            'null_r_distribution_std': null_std,
            'null_z_score': z,
            'null_percentile': percentile,
            'n_null_draws': n_draws,
        }

    compute_null_stats.redraw_totals = redraw_totals
    return compute_null_stats


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument('--n-null-draws', type=int, default=DEFAULT_N_NULL_DRAWS,
                         help=f'Independent cross-song substitutions drawn per test trial at '
                              f'eval time to build the null r distribution (default: '
                              f'{DEFAULT_N_NULL_DRAWS}; shrunk under --smoke-test).')
    args = parser.parse_args(sys.argv[1:])
    args.variants = args.variants if '--variants' in sys.argv else 'nonlinear'
    config = load_config(cli_args=sys.argv[1:])
    manifest = build_manifest(config)

    n_null_draws = SMOKE_TEST_N_NULL_DRAWS if args.smoke_test else args.n_null_draws

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
    pairing_table_holder = {}

    def make_window_dataset_cls(ds_by_subject, feature_keys, fold):
        pairing_table = NullPairingTable(
            ds_by_subject, manifest,
            sorted(set(fold.train_keys) | set(fold.val_keys) | set(fold.test_keys)),
            feature_keys, config.sfreq, seed=seed_holder['seed'] + fold.fold_idx)
        pairing_table_holder[fold.fold_idx] = pairing_table
        return partial(MismatchedMultiSubjectWindowDataset, pairing_table=pairing_table)

    compute_null_stats = make_compute_null_stats(
        manifest, config.sfreq, n_null_draws, base_seed=seed_holder['seed'])

    saved_paths = run_all_folds(
        PARADIGM_NAME, folds, config, cache, args, model_family='conv_null',
        extra_meta_base={'null_baseline': True, 'mismatch_strategy': MISMATCH_STRATEGY_DESC},
        make_window_dataset_cls=make_window_dataset_cls,
        compute_null_stats=compute_null_stats,
    )

    if args.smoke_test:
        for fold_idx, table in pairing_table_holder.items():
            summary = table.summary()
            assert summary['n_pairs'] > 0, f"fold {fold_idx}: no mismatched pairs built"
            print(f"[null_baseline] fold {fold_idx} mismatch summary: {summary}")
        print(f"[null_baseline] mismatch-verification asserts passed for every sampled pair "
              f"({compute_null_stats.redraw_totals['total']} redraws across "
              f"{compute_null_stats.redraw_totals['pairs']} null-eval draws, "
              f"n_null_draws={n_null_draws})")

        assert saved_paths, "[null_baseline] smoke test produced no output pickles"
        for path in saved_paths:
            with open(path, 'rb') as f:
                result = pickle.load(f)
            meta = result['meta']
            assert meta.get('null_r_distribution_mean') is not None, \
                f"{path}: missing null distribution stats in meta"
            assert meta['null_r_distribution_mean'].shape == result['r_per_channel'].shape, \
                f"{path}: null distribution shape mismatch"
            assert meta.get('null_r_all_draws') is not None \
                and meta['null_r_all_draws'].shape[0] == meta['n_null_draws'], \
                f"{path}: null_r_all_draws shape/count mismatch"
        print(f"[null_baseline] null-distribution wiring verified for {len(saved_paths)} pickles")


if __name__ == '__main__':
    main()
