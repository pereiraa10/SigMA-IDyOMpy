"""
run_common.py — shared "train + evaluate + save" loop for one Fold, plus the
--smoke-test / --fold CLI plumbing, reused by every run_paradigmN_*.py entry
script and null_baseline_mismatched_stimulus.py. The only thing that differs
between paradigms is *how* a Fold is built (splits.py); once a Fold exists,
running it is identical, so that loop lives here once instead of copy-pasted
four times.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, not a package

import results as res
from TRF_conv import (
    extract_kernel, WINDOW_SAMPLES, HOP_SAMPLES,
    EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE, HIDDEN, N_BLOCKS, BATCH_SIZE, SEED,
)

from dataset_pool import MultiSubjectWindowDataset
from training_harness import train_pooled, evaluate_full_trials
from results_io import BASE_RESULTS_DIR, fold_dir, save_holdout_result
from splits import Fold, log_fold_manifest

# Mirrors TRF_conv.py's `hyperparams` dict (its main()), so every pickle logs
# the FULL set of hyperparameters actually used -- defaults included, not
# just whatever was overridden via train_kwargs/CLI flags.
_DEFAULT_TRAIN_KWARGS = dict(
    epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, patience=EARLY_STOP_PATIENCE,
    hidden=HIDDEN, n_blocks=N_BLOCKS, batch_size=BATCH_SIZE, seed=SEED,
)


def build_arg_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--smoke-test', action='store_true',
                    help='Restrict to 1 fold, <=2 trials/subject per split, 3 epochs, '
                         'patience=2, batch_size=8, one (feature_set, variant) combo, '
                         'and write to results/_smoke_test/ instead of the real output dir.')
    p.add_argument('--fold', type=int, default=None,
                    help='Restrict to a single fold index (default: run every fold).')
    p.add_argument('--feature-sets', default=None,
                    help='Comma-separated feature_set subset (default: every feature_set in config).')
    p.add_argument('--variants', default='linear,nonlinear',
                    help="Comma-separated StimToEEG variants to run (default: 'linear,nonlinear').")
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--no-plots', action='store_true', help='Skip alignment/topo PNG generation.')
    return p


def shrink_fold_for_smoke_test(fold, max_per_subject=2, max_test_subjects=2,
                                max_val_subjects=2, max_train_subjects=4):
    """Cap BOTH the number of distinct subjects touched (per role) AND the
    trials/subject within each split, so SubjectCache only builds
    PreparedSubject for a handful of subjects total -- fast wiring check,
    not a real training run.

    Capping subjects per role (not just trials/subject) matters for
    paradigms 1/2, where train_subjects == val_subjects == test_subjects
    (every subject participates in every role) -- without this, a
    trials-per-subject-only cap would still touch all 20 subjects' full
    preprocessing pipeline. For paradigms 3/4 (disjoint role subject sets)
    this also naturally preserves at least a few train/val/test subjects
    each, so training still has non-empty train_keys/val_keys.
    """
    keep_test = list(dict.fromkeys(fold.test_subjects))[:max_test_subjects]
    keep_val = list(dict.fromkeys(fold.val_subjects))[:max_val_subjects]
    keep_train = list(dict.fromkeys(fold.train_subjects))[:max_train_subjects]
    keep_subjects = set(keep_test) | set(keep_val) | set(keep_train)

    def _cap(keys):
        seen = {}
        out = []
        for s, i in keys:
            if s not in keep_subjects:
                continue
            n = seen.get(s, 0)
            if n < max_per_subject:
                out.append((s, i))
                seen[s] = n + 1
        return out

    train_keys, val_keys, test_keys = _cap(fold.train_keys), _cap(fold.val_keys), _cap(fold.test_keys)

    tr, va, te = set(train_keys), set(val_keys), set(test_keys)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), \
        "shrink_fold_for_smoke_test: introduced leakage while shrinking"
    assert train_keys and val_keys, \
        "shrink_fold_for_smoke_test: shrunk fold has an empty train or val split"

    def _present(subjects, keys):
        present = {s for s, _ in keys}
        return [s for s in subjects if s in present]

    return Fold(
        fold.paradigm, fold.fold_idx, train_keys, val_keys, test_keys,
        train_subjects=_present(fold.train_subjects, train_keys),
        val_subjects=_present(fold.val_subjects, val_keys),
        test_subjects=_present(fold.test_subjects, test_keys),
        train_songs=fold.train_songs, val_songs=fold.val_songs, test_songs=fold.test_songs,
    )


def run_fold(fold, config, cache, feature_sets, variants, base_results_dir,
             window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES,
             window_dataset_cls=MultiSubjectWindowDataset, make_window_dataset_cls=None,
             model_family='conv', extra_meta_base=None, plot=True, debug=False,
             compute_null_stats=None, **train_kwargs):
    """Train + evaluate + save pickles for one Fold, across every
    (feature_set, variant) combination. Returns list of saved paths.

    make_window_dataset_cls : optional callable(ds_by_subject, feature_keys,
        fold) -> a MultiSubjectWindowDataset-shaped class, called once per
        feature_set (before the variant loop) to override `window_dataset_cls`.
        Used by null_baseline_mismatched_stimulus.py to build a
        feature_set-specific mismatched-pairing table and hand back a bound
        MismatchedMultiSubjectWindowDataset without duplicating this loop.

    compute_null_stats : optional callable(model, ds_by_subject, feature_keys,
        test_subject, subj_test_keys, Y_true, Y_pred, trial_boundaries,
        channel_names, fold, feature_set, variant) -> dict, called once per
        (feature_set, variant, test_subject) right after evaluate_full_trials,
        with the result merged into that result's extra_meta (and therefore
        into the saved pickle's meta dict). Used by
        null_baseline_mismatched_stimulus.py to attach a permutation-style
        null r distribution to each result without retraining the model.
    """
    out_dir = fold_dir(fold, base_results_dir)
    log_fold_manifest(fold, out_path=out_dir / f'{fold.paradigm}_fold{fold.fold_idx}_manifest.txt')

    all_keys = sorted(set(fold.train_keys) | set(fold.val_keys) | set(fold.test_keys))
    effective_train_kwargs = {**_DEFAULT_TRAIN_KWARGS, **train_kwargs}
    saved_paths = []
    for feature_set in feature_sets:
        feature_keys = config.feature_sets[feature_set]
        ds_by_subject = cache.datasets_for_keys(all_keys, feature_set, window_samples, hop_samples)
        n_channels = next(iter(ds_by_subject.values())).n_channels

        this_window_dataset_cls = (
            make_window_dataset_cls(ds_by_subject, feature_keys, fold)
            if make_window_dataset_cls is not None else window_dataset_cls)

        for variant in variants:
            model, tr_mse, v_mse, v_r = train_pooled(
                ds_by_subject, fold.train_keys, fold.val_keys,
                n_features=len(feature_keys), n_channels=n_channels, variant=variant,
                window_dataset_cls=this_window_dataset_cls, debug=debug, **train_kwargs)
            weights = extract_kernel(model, variant)
            training_history = {'train_mse': tr_mse, 'val_mse': v_mse, 'val_r': v_r}

            for test_subject in fold.test_subjects:
                subj_test_keys = sorted(k for k in fold.test_keys if k[0] == test_subject)
                if not subj_test_keys:
                    continue
                Y_pred, Y_true, trial_boundaries = evaluate_full_trials(
                    model, subj_test_keys, ds_by_subject, feature_keys)

                extra_meta = {
                    'paradigm': fold.paradigm, 'fold': fold.fold_idx,
                    'train_subjects': fold.train_subjects, 'val_subjects': fold.val_subjects,
                    'test_subjects': fold.test_subjects,
                    'train_songs': fold.train_songs, 'val_songs': fold.val_songs,
                    'test_songs': fold.test_songs,
                    'hyperparams': {**effective_train_kwargs, 'variant': variant,
                                    'window_samples': window_samples, 'hop_samples': hop_samples},
                }
                if extra_meta_base:
                    extra_meta.update(extra_meta_base)

                if compute_null_stats is not None:
                    null_meta = compute_null_stats(
                        model=model, ds_by_subject=ds_by_subject, feature_keys=feature_keys,
                        test_subject=test_subject, subj_test_keys=subj_test_keys,
                        Y_true=Y_true, Y_pred=Y_pred, trial_boundaries=trial_boundaries,
                        channel_names=ds_by_subject[test_subject].channel_names,
                        fold=fold, feature_set=feature_set, variant=variant)
                    extra_meta.update(null_meta)

                result = res.build_result(
                    subject=test_subject, subject_type=config.subject_type[test_subject],
                    feature_set=feature_set, feature_keys=feature_keys, model_family=model_family,
                    model_variant=variant, channel_names=ds_by_subject[test_subject].channel_names,
                    Y_true=Y_true, Y_pred=Y_pred, trial_boundaries=trial_boundaries,
                    weights=weights, training_history=training_history, extra_meta=extra_meta,
                )
                path = res.result_filename(out_dir, test_subject, model_family, feature_set, variant=variant)
                save_holdout_result(path, result, plot=plot)
                saved_paths.append(path)
                print(f"  [{fold.paradigm} fold{fold.fold_idx}] {test_subject} | {feature_set} | "
                      f"{model_family}_{variant}: mean r = {result['r_per_channel'].mean():.4f}")
    return saved_paths


def run_all_folds(paradigm_name, folds, config, cache, args, model_family='conv',
                   extra_meta_base=None, window_dataset_cls=MultiSubjectWindowDataset,
                   make_window_dataset_cls=None, compute_null_stats=None):
    """Shared driver for a run_paradigmN_*.py __main__ block: applies
    --smoke-test/--fold/--feature-sets/--variants/--epochs/--patience/
    --batch-size, then calls run_fold for each selected fold."""
    if args.fold is not None:
        folds = [f for f in folds if f.fold_idx == args.fold]
        if not folds:
            raise ValueError(f"--fold {args.fold} not found for {paradigm_name}")

    feature_sets = (args.feature_sets.split(',') if args.feature_sets
                     else list(config.feature_sets))
    variants = args.variants.split(',')

    train_kwargs = {}
    if args.epochs is not None:
        train_kwargs['epochs'] = args.epochs
    if args.patience is not None:
        train_kwargs['patience'] = args.patience
    if args.batch_size is not None:
        train_kwargs['batch_size'] = args.batch_size

    base_results_dir = BASE_RESULTS_DIR
    if args.smoke_test:
        folds = [shrink_fold_for_smoke_test(folds[0])]
        feature_sets = feature_sets[:1]
        variants = variants[:1]
        train_kwargs = {'epochs': 3, 'patience': 2, 'batch_size': 8}
        base_results_dir = BASE_RESULTS_DIR / '_smoke_test'
        print(f"[{paradigm_name}] --smoke-test: 1 fold, <=2 trials/subject/split, "
              f"feature_sets={feature_sets}, variants={variants}, {train_kwargs}")

    all_saved = []
    for fold in folds:
        saved = run_fold(
            fold, config, cache, feature_sets, variants, base_results_dir,
            model_family=model_family, extra_meta_base=extra_meta_base,
            window_dataset_cls=window_dataset_cls, make_window_dataset_cls=make_window_dataset_cls,
            compute_null_stats=compute_null_stats,
            plot=not args.no_plots, debug=args.smoke_test,
            **train_kwargs)
        all_saved += saved

    if args.smoke_test:
        _smoke_test_checks(paradigm_name, folds, all_saved)
    return all_saved


def _smoke_test_checks(paradigm_name, folds, saved_paths):
    """Post-hoc assertions proving the smoke-test run actually wired
    together correctly: pickle round-trip + expected fields present."""
    import pickle
    assert saved_paths, f"[{paradigm_name}] smoke test produced no output pickles"
    for path in saved_paths:
        assert path.exists(), f"[{paradigm_name}] {path} was not written"
        with open(path, 'rb') as f:
            result = pickle.load(f)
        meta = result['meta']
        assert meta['subject'], f"{path}: missing meta['subject']"
        assert result['r_per_channel'].shape == (len(meta['channel_names']),), \
            f"{path}: r_per_channel shape mismatch"
        assert meta.get('paradigm') and meta.get('fold') is not None, \
            f"{path}: missing paradigm/fold in meta (extra_meta wiring broken)"
        assert not any(_isnan(x) for x in result['training_history']['val_r']), \
            f"{path}: NaN in val_r history"
    print(f"[{paradigm_name}] SMOKE TEST PASSED ({len(saved_paths)} pickles round-tripped OK)")


def _isnan(x):
    return x != x
