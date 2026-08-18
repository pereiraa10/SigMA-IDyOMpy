"""
results.py — shared pickle output schema and filename convention for the TRF
experiment scripts in experiments/.

This intentionally contains no model-fitting or evaluation logic (see
EVALUATION_NOTES.md for that — LOOCV harnesses, alpha selection, and
lag-matrix construction stay in each script since they differ per model
family). It only standardizes what gets written to disk once a script has
already produced its own Y_pred/Y_true/weights, so every pickle in
the run's save_dir has the same shape regardless of which script produced it.
That's what makes cross-model notebook analysis (avg_trf.ipynb,
correlation_plot.ipynb) possible without per-script loading code.

Filename convention
--------------------
    {subject}__{model_family}[_{variant}]__{feature_set}.pkl
e.g. Sub10__sklearn_ridge__acoustic.pkl
     Sub10__conv__nonlinear__acoustic_and_surprisal.pkl   (variant kept distinct)

Replaces the old `Sub10_['envelope', 'onsets']_sklearn_ridge_acoustic_data.pkl`
scheme, which embedded the raw Python list repr of feature_keys in the
filename. feature_keys now live in meta['feature_keys'] instead.

Pickle schema
-------------
    {
      'meta': {
          'subject', 'subject_type', 'feature_set', 'feature_keys',
          'model_family', 'model_variant', 'channel_names',
          'trial_boundaries',   # list[(start, end)] into Y_pred/Y_true per trial
          'best_alpha', 'date_run', ...any extra_meta passed in
      },
      'r_per_channel':    (n_channels,) ndarray — whole-session concatenated-holdout r
      'per_trial_r':      (n_trials, n_channels) ndarray
      'Y_pred', 'Y_true': (T_total, n_channels) float32 ndarray — always saved,
                          every model. float32 (not float64): a full-precision
                          pair runs ~300MB/file, which multiplies to 50-100+ GB
                          across 20 subjects x 2 feature_sets x ~6 model/variants;
                          float32 halves that and is still far finer than the
                          64 Hz EEG signal's actual precision. Correlations in
                          this file are computed before downcasting.
      'weights':          (n_channels, n_lags, n_features) ndarray or None
      'alpha_selection':  dict {alpha: mean_cv_r} or None   (ridge only)
      'training_history': dict of per-epoch arrays or None (conv only)
    }
"""
import pickle
from datetime import date

import numpy as np
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt
import eelbrain

DEFAULT_SFREQ = 64
DEFAULT_CHANNEL_IDX = 0

# Mirrors the tmin/tmax receptive-field convention shared by every config.yaml
# in this project (-0.1 / 0.6 s, i.e. a 700 ms window) -- used as the ERP
# window length for the grand-average topo plots below, not as a literal
# pre-stimulus slice (see _grand_average_ndvars).
DEFAULT_TMIN = -0.1
DEFAULT_TMAX = 0.600

# Tried in order to reconstruct sensor positions from meta['channel_names']
# alone (no montage/positions are saved in the pickle). biosemi64 is
# confirmed correct for the OpenMIIR dataset (config_openmiir.yaml); the
# others cover datasets loaded via _load_eeg_from_edf/_load_eeg_from_mat
# whose channel naming hasn't been verified against a standard montage.
_CANDIDATE_MONTAGES = ['biosemi64', 'standard_1020', 'standard_1005']
_TOPOARRAY_TIME_FRACTIONS = (0.15, 0.5, 0.85)  # fractions of window duration


def _reload_raw_eeg_trials(subject):
    """Reload `subject`'s preprocessed-but-NOT-z-scored EEG trials (pipeline
    steps 1-6: LPF -> downsample -> HPF -> strip padding, no step-7 z-scoring)
    straight from disk, for recovering raw-microvolt units in plots.

    The z-scored Y_true/Y_pred saved in the pickle carry no record of the
    per-trial mean/std that produced them (see module docstring: the schema
    is intentionally just meta/r/Y_pred/Y_true/weights/...), so there is
    nothing to invert with from the pickle alone. This re-runs config.yaml's
    same subject-level load + steps 1-6 (deterministic, not affected by
    feature_set) to recompute those statistics on demand.

    This is the one deliberate exception to this module's no-pipeline-
    dependency design (see module docstring) -- only reached when a caller
    passes raw_units=True, and it requires the same raw EEG on disk (and the
    same config.yaml) that produced the pickle in the first place.
    """
    from config import load_config
    import utils as pipeline_utils
    from dataset import PreparedSubject

    config = load_config()
    eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=subject)
    eeg_data = pipeline_utils.load_subject_raw_eeg(
        eeg_path, subject, config.trial_to_stimulus.get(subject))
    prepared = PreparedSubject(subject, eeg_data, config)
    return prepared._trials_raw


def _invert_zscore(Y, trial_boundaries, subject):
    """Map a z-scored (T_total, n_channels) array (Y_true or Y_pred) back to
    raw microvolts, trial by trial: raw = z * std + mean, using each trial's
    own mean/std recomputed from the un-z-scored EEG (see
    _reload_raw_eeg_trials). Y_pred is inverted with the SAME per-trial
    mean/std as Y_true (its own trial's ground-truth statistics), since Y_pred
    is a prediction of that trial's z-scored EEG, not an independently scaled
    quantity.

    Note this is a linear transform: it restores physically interpretable
    units but does not change the relative amplitude gap between Y_pred and
    Y_true (that gap comes from prediction quality, i.e. r), so a weak model's
    prediction will still look small next to the actual signal in raw units.
    """
    raw_trials = _reload_raw_eeg_trials(subject)
    if len(raw_trials) != len(trial_boundaries):
        raise ValueError(
            f"{subject}: reloaded {len(raw_trials)} raw trials but the "
            f"pickle has {len(trial_boundaries)} trial_boundaries -- raw EEG "
            f"on disk (or config.yaml) no longer matches what produced this "
            f"result. Can't safely invert z-scoring."
        )

    out = np.empty_like(Y)
    for (start, end), raw_t in zip(trial_boundaries, raw_trials):
        eeg = raw_t['eeg']
        if eeg.shape[0] != end - start:
            raise ValueError(
                f"{subject}: raw trial length {eeg.shape[0]} != pickled "
                f"trial length {end - start} for boundary ({start}, {end}). "
                f"Can't safely invert z-scoring."
            )
        mean, std = eeg.mean(axis=0), eeg.std(axis=0)
        out[start:end] = Y[start:end] * std + mean
    return out


def result_filename(save_dir, subject, model_family, feature_set, variant=None):
    model_tag = model_family if variant is None else f'{model_family}_{variant}'
    return save_dir / f'{subject}__{model_tag}__{feature_set}.pkl'


def per_trial_r(Y_true, Y_pred, trial_boundaries):
    """Per-trial, per-channel Pearson r. Returns (n_trials, n_channels)."""
    n_channels = Y_true.shape[1]
    out = np.zeros((len(trial_boundaries), n_channels))
    for ti, (start, end) in enumerate(trial_boundaries):
        for ch in range(n_channels):
            out[ti, ch] = pearsonr(Y_true[start:end, ch], Y_pred[start:end, ch])[0]
    return out


def build_result(*, subject, subject_type, feature_set, feature_keys, model_family,
                  channel_names, Y_true=None, Y_pred=None, trial_boundaries=None,
                  r_per_channel=None, model_variant=None, best_alpha=None,
                  alpha_selection=None, weights=None, training_history=None,
                  extra_meta=None):
    """Assemble the standardized result dict every script pickles.

    Usual case (sklearn/mne/conv)
    ------------------------------
    Y_true, Y_pred : (T_total, n_channels) ndarray
        Concatenated held-out predictions/actual, in trial order.
    trial_boundaries : list[(start, end)]
        Index ranges into Y_true/Y_pred per trial, so per-trial slices (for
        alignment plots, per-trial r, etc.) don't require re-running the model.
    r_per_channel is computed from Y_true/Y_pred automatically when both are given.

    Exception (eelbrain.boosting)
    ------------------------------
    eelbrain.boosting's cross-validated predictions aren't exposed as a plain,
    trial-aligned array (see EVALUATION_NOTES.md), so TRF_pickle_A_and_AM.py
    passes Y_true=Y_pred=None and supplies `r_per_channel` directly from
    `trf_cv.r`. In that case `per_trial_r` is left as None.

    Optional
    --------
    best_alpha, alpha_selection : ridge only.
        alpha_selection is a dict {alpha: mean_cv_r} from the LOOCV alpha
        search — previously computed then discarded without being saved.
    weights : (n_channels, n_lags, n_features) ndarray or None.
        The TRF kernel, where the model family has one in that shape (all
        ridge variants; conv 'linear'/'separable' variants, since their first
        conv layer *is* the TRF kernel).
    training_history : dict of per-epoch arrays, conv models only.
    """
    pt_r = None
    if Y_true is not None and Y_pred is not None:
        n_channels = Y_true.shape[1]
        computed_r = np.array([
            pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(n_channels)
        ])
        r_per_channel = computed_r if r_per_channel is None else r_per_channel
        if trial_boundaries is not None:
            pt_r = per_trial_r(Y_true, Y_pred, trial_boundaries)
    elif r_per_channel is None:
        raise ValueError(
            "build_result needs either (Y_true and Y_pred) or an explicit "
            "r_per_channel override.")

    # Cross-check channel labeling against the prediction array width, so a
    # mismatched channel list can't silently mislabel per-channel results.
    # Skipped when Y_true is None (the boosting-without-reconstruction case).
    if Y_true is not None and channel_names is not None:
        if len(channel_names) != Y_true.shape[1]:
            raise ValueError(
                f"channel_names length ({len(channel_names)}) != Y_true channel "
                f"count ({Y_true.shape[1]}) for {subject} / {feature_set} / "
                f"{model_family}."
            )

    meta = {
        'subject': subject,
        'subject_type': subject_type,
        'feature_set': feature_set,
        'feature_keys': list(feature_keys),
        'model_family': model_family,
        'model_variant': model_variant,
        'channel_names': list(channel_names),
        'trial_boundaries': trial_boundaries,
        'best_alpha': best_alpha,
        'date_run': str(date.today()),
    }
    if extra_meta:
        meta.update(extra_meta)

    # Correlations above are computed from the original (float64) arrays for
    # numerical accuracy; only the stored copies are downcast. A ~300MB/file
    # float64 Y_pred+Y_true pair (20 subjects x 2 feature_sets x ~6 models would
    # be 50-100+ GB) halves to ~150MB in float32, which is still far finer
    # than the 64 Hz EEG signal's actual precision.
    if Y_pred is not None:
        Y_pred = Y_pred.astype(np.float32)
    if Y_true is not None:
        Y_true = Y_true.astype(np.float32)

    return {
        'meta': meta,
        'r_per_channel': r_per_channel,
        'per_trial_r': pt_r,
        'Y_pred': Y_pred,
        'Y_true': Y_true,
        'weights': weights,
        'alpha_selection': alpha_selection,
        'training_history': training_history,
    }


def plot_alignment(result, save_dir, channel_idx=DEFAULT_CHANNEL_IDX, sfreq=DEFAULT_SFREQ,
                    raw_units=True):
    """Predicted-vs-actual EEG alignment plot, built entirely from a `result`
    dict (see build_result). Skipped when the result has no Y_true/Y_pred
    (the eelbrain.boosting exception, see build_result's docstring).

    raw_units : if True, invert the per-trial z-scoring (see
        _invert_zscore) so both traces are plotted in raw microvolts instead
        of z-score units. Requires the original raw EEG + config.yaml to
        still be reachable on disk (see _reload_raw_eeg_trials)."""
    Y_true, Y_pred = result['Y_true'], result['Y_pred']
    if Y_true is None or Y_pred is None:
        return

    meta = result['meta']
    subject, feature_set = meta['subject'], meta['feature_set']
    model_tag = meta['model_family']
    if meta.get('model_variant'):
        model_tag = f"{model_tag}_{meta['model_variant']}"

    if raw_units:
        trial_boundaries = meta['trial_boundaries']
        Y_true = _invert_zscore(Y_true, trial_boundaries, subject)
        Y_pred = _invert_zscore(Y_pred, trial_boundaries, subject)
        unit_label, actual_label = 'μV', 'Actual EEG (raw μV)'
    else:
        unit_label, actual_label = 'z-score', 'Actual EEG (z-scored)'

    r_vals = result['r_per_channel']
    ch = channel_idx if channel_idx < Y_true.shape[1] else 0
    n_plot = min(len(Y_true), int(10 * sfreq))
    t_plot = np.arange(n_plot) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f'Predicted vs Actual EEG  |  {subject}, channel {ch}\n'
        f'{model_tag}  ·  {feature_set}  ·  r = {r_vals[ch]:.3f}',
        fontsize=12, fontweight='bold')

    axes[0].plot(t_plot, Y_true[:n_plot, ch],
                 color='black', lw=0.7, label=actual_label)
    axes[0].plot(t_plot, Y_pred[:n_plot, ch],
                 color='seagreen', lw=0.9, alpha=0.85,
                 label=f'Predicted EEG  (r = {r_vals[ch]:.3f})')
    axes[0].set_ylabel(unit_label)
    axes[0].set_title(f'Actual vs Predicted EEG  ({model_tag})')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_plot, Y_true[:n_plot, ch] - Y_pred[:n_plot, ch],
                 color='darkorange', lw=0.7, label='Residual (actual - predicted)')
    axes[1].axhline(0, color='black', lw=0.6, linestyle='--')
    axes[1].set_ylabel(unit_label)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Residual: Actual - Predicted')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = '_raw' if raw_units else ''
    fname = save_dir / f"{subject}_{feature_set}_{model_tag}_alignment_ch{ch}{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _build_sensor(channel_names):
    """eelbrain.Sensor reconstructed from channel *names* alone (the pickle
    schema doesn't save positions/montage) by matching against a short list
    of candidate standard montages. Raises if none of them cover every name
    in `channel_names` -- add the dataset's real montage to
    _CANDIDATE_MONTAGES rather than silently guessing."""
    errors = []
    for montage in _CANDIDATE_MONTAGES:
        try:
            return eelbrain.Sensor.from_montage(montage, channels=channel_names)
        except ValueError as e:
            errors.append(f"{montage}: {e}")
    raise ValueError(
        f"None of the candidate montages {_CANDIDATE_MONTAGES} contain all "
        f"channel names {channel_names}. Add the correct montage to "
        f"_CANDIDATE_MONTAGES in results.py. Errors: {errors}"
    )


def _build_case_ndvars(result, sfreq, tmin=DEFAULT_TMIN, tmax=DEFAULT_TMAX, raw_units=True,
                        onsets=None):
    """Y_true/Y_pred for every trial, as a pair of (case, sensor, time)
    NDVars, not pre-averaged. eelbrain.plot.TopoButterfly / TopoArray average
    over the Case dimension themselves when rendering, so no manual averaging
    happens here.

    Two slicing modes, selected by `onsets`:

    onsets=None (default) -- one Case per trial. Each trial is truncated to
        its first (tmax - tmin) seconds (700 ms by default, matching this
        project's TRF tmin/tmax receptive-field convention) so every trial
        contributes a NDVar of the same length; every trial in this dataset
        is far longer than that, so this never runs past a trial's end.
        Windows start at 0 (trial/stimulus onset), not tmin: the saved
        Y_true/Y_pred arrays only cover each trial's own span (see
        build_trials), so there's no genuine pre-stimulus baseline to slice
        for tmin < 0 -- indices before a trial's start in the concatenated
        array belong to a different trial (or don't exist, for the first
        trial). (tmax - tmin) is used purely as the window *length*;
        tmin/tmax themselves are passed to the plot functions' `xlim` so the
        displayed time axis still reads in the -100 ms .. +600 ms convention.

    onsets=list[array-like] (len == n_trials) -- one Case per onset EVENT,
        pooled across all trials, for a genuine onset-locked grand-average
        ERP. onsets[i] is a 1-D array of sample indices (relative to trial
        i's own start, at `sfreq`) marking each onset event to lock a window
        to -- e.g. `np.flatnonzero(trial['onset_surprisal'])` for the sparse
        IDyOM-onset impulses, or a peak-picked/thresholded index list derived
        from the continuous acoustic `trial['onsets']` envelope-derivative
        signal. Each window is [onset + tmin*sfreq, onset + tmin*sfreq +
        (tmax-tmin)*sfreq) samples, so it DOES include genuine pre-onset
        samples (unlike the trial-start mode above). Events whose window
        would fall outside their trial's bounds are dropped rather than
        clipped, so every Case has the identical, real tmin..tmax span. The
        time axis is anchored at tmin (not 0), since t=0 now means the actual
        onset for every Case.

    raw_units : if True, invert the per-trial z-scoring (over the FULL trial,
        before either windowing mode above) so the NDVars carry raw
        microvolts instead of z-score units -- see _invert_zscore.

    Returns None (skip) under the same condition plot_alignment skips under
    -- no Y_true/Y_pred (the eelbrain.boosting exception) -- or if
    trial_boundaries wasn't saved. Raises ValueError if `onsets` doesn't
    have one entry per trial, or if no onset event's window fits inside its
    trial's bounds.

    Returns (nd_true, nd_pred, meta, model_tag).
    """
    Y_true, Y_pred = result['Y_true'], result['Y_pred']
    trial_boundaries = result['meta'].get('trial_boundaries')
    if Y_true is None or Y_pred is None or trial_boundaries is None:
        return None

    meta = result['meta']
    model_tag = meta['model_family']
    if meta.get('model_variant'):
        model_tag = f"{model_tag}_{meta['model_variant']}"

    if raw_units:
        Y_true = _invert_zscore(Y_true, trial_boundaries, meta['subject'])
        Y_pred = _invert_zscore(Y_pred, trial_boundaries, meta['subject'])

    sensor = _build_sensor(meta['channel_names'])

    if onsets is None:
        n_window = min(
            round((tmax - tmin) * sfreq),
            min(end - start for start, end in trial_boundaries))
        time_axis = eelbrain.UTS(0, 1 / sfreq, n_window)

        def case_ndvar(Y, name):
            arrs = np.stack([Y[start:start + n_window] for start, _ in trial_boundaries])
            return eelbrain.NDVar(
                arrs.transpose(0, 2, 1), (eelbrain.Case, sensor, time_axis), name=name)
    else:
        if len(onsets) != len(trial_boundaries):
            raise ValueError(
                f"onsets has {len(onsets)} entries but result has "
                f"{len(trial_boundaries)} trials -- need exactly one "
                f"per-trial onset-index array (empty is fine for a trial "
                f"with no events)."
            )
        n_window = round((tmax - tmin) * sfreq)
        pre_samples = round(-tmin * sfreq)
        time_axis = eelbrain.UTS(tmin, 1 / sfreq, n_window)

        # Window bounds (in trial-local samples) don't depend on Y_true vs
        # Y_pred, so compute them once and reuse for both.
        windows = []  # list of (start, end) into the concatenated Y arrays
        for (t_start, t_end), trial_onsets in zip(trial_boundaries, onsets):
            trial_len = t_end - t_start
            for on in trial_onsets:
                w_start = int(on) - pre_samples
                w_end = w_start + n_window
                if w_start < 0 or w_end > trial_len:
                    continue
                windows.append((t_start + w_start, t_start + w_end))
        if not windows:
            raise ValueError(
                "No onset-locked window fit inside its trial's bounds for "
                f"tmin={tmin}, tmax={tmax} -- check that `onsets` indices "
                "are trial-local sample positions at the same sfreq as the "
                "result, and that events aren't all too close to a trial "
                "edge for this window."
            )

        def case_ndvar(Y, name):
            arrs = np.stack([Y[start:end] for start, end in windows])
            return eelbrain.NDVar(
                arrs.transpose(0, 2, 1), (eelbrain.Case, sensor, time_axis), name=name)

    nd_true = case_ndvar(Y_true, 'Actual')
    nd_pred = case_ndvar(Y_pred, 'Predicted')
    return nd_true, nd_pred, meta, model_tag


def plot_topobutterfly(result, save_dir, sfreq=DEFAULT_SFREQ,
                        tmin=DEFAULT_TMIN, tmax=DEFAULT_TMAX, raw_units=False,
                        onsets=None, onset_label=None):
    """Actual-vs-predicted ERP as an eelbrain TopoButterfly: all channels
    overlaid, plus a scalp topomap. Feeds Y_true/Y_pred to TopoButterfly
    directly (as a Case dimension) and lets it average over Cases and pick
    the topomap time itself -- no manual averaging or peak-time computation
    here. tmin/tmax set the displayed window (xlim), matching this project's
    TRF receptive-field convention.

    onsets : optional list (len == n_trials) of per-trial onset sample-index
        arrays. When given, the grand average is locked to each onset EVENT
        (pooled across all trials) instead of one Case per trial-start --
        see _build_case_ndvars for the exact windowing rule. Use this to
        compare, e.g., acoustic-onset-locked vs IDyOM-onset-surprisal-locked
        grand averages against each other.
    onset_label : optional str tag (e.g. 'onset_surprisal') describing the
        onset source in `onsets`, folded into the title/filename so runs
        with different onset sources don't overwrite each other. Ignored if
        onsets is None.

    raw_units : if True, plot raw microvolts instead of z-score units -- see
        _invert_zscore / _build_case_ndvars."""
    built = _build_case_ndvars(result, sfreq, tmin=tmin, tmax=tmax, raw_units=raw_units,
                                onsets=onsets)
    if built is None:
        return
    nd_true, nd_pred, meta, model_tag = built
    subject, feature_set = meta['subject'], meta['feature_set']

    title = f'{subject}  ·  {model_tag}  ·  {feature_set}' + ('  ·  raw μV' if raw_units else '')
    suffix = '_raw' if raw_units else ''
    if onsets is not None:
        n_events = nd_true.shape[0]
        tag = onset_label or 'onset-locked'
        title += f'  ·  {tag}-locked (n={n_events})'
        suffix += f'_{onset_label}' if onset_label else '_onsetlocked'
    p = eelbrain.plot.TopoButterfly(
        [nd_true, nd_pred], 
        xlim=(tmin, tmax), 
        w=12, 
        h=6, 
        clip='circle',
        show=False, 
        title=title, 
        xticklabels = 'bottom',
        yticklabels = 'bottom'
        )
    fname = save_dir / f"{subject}_{feature_set}_{model_tag}_topobutterfly{suffix}.png"
    p.save(fname)
    p.close()
    return fname


def plot_topoarray(result, save_dir, sfreq=DEFAULT_SFREQ,
                    tmin=DEFAULT_TMIN, tmax=DEFAULT_TMAX, raw_units=True,
                    onsets=None, onset_label=None):
    """Actual-vs-predicted ERP as an eelbrain TopoArray: scalp topomaps at a
    few representative times spread across the tmin/tmax window. Feeds
    Y_true/Y_pred to TopoArray directly (as a Case dimension) and lets it
    average over Cases -- no manual averaging here.

    onsets, onset_label : see plot_topobutterfly -- same onset-locked
        grand-average mode, forwarded to _build_case_ndvars unchanged.

    raw_units : if True, plot raw microvolts instead of z-score units -- see
        _invert_zscore / _build_case_ndvars."""
    built = _build_case_ndvars(result, sfreq, tmin=tmin, tmax=tmax, raw_units=raw_units,
                                onsets=onsets)
    if built is None:
        return
    nd_true, nd_pred, meta, model_tag = built
    subject, feature_set = meta['subject'], meta['feature_set']

    times = [tmin + f * (tmax - tmin) for f in _TOPOARRAY_TIME_FRACTIONS]
    title = f'{subject}  ·  {model_tag}  ·  {feature_set}' + ('  ·  raw μV' if raw_units else '')
    suffix = '_raw' if raw_units else ''
    if onsets is not None:
        n_events = nd_true.shape[0]
        tag = onset_label or 'onset-locked'
        title += f'  ·  {tag}-locked (n={n_events})'
        suffix += f'_{onset_label}' if onset_label else '_onsetlocked'
    p = eelbrain.plot.TopoArray(
        [nd_true, nd_pred], t=times, xlim=(tmin, tmax), w=12, h=6, clip='circle',
        show=False, title=title)
    fname = save_dir / f"{subject}_{feature_set}_{model_tag}_topoarray{suffix}.png"
    p.save(fname)
    p.close()
    return fname


def save_result(path, result, channel_idx=DEFAULT_CHANNEL_IDX, sfreq=DEFAULT_SFREQ):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    plot_alignment(result, path.parent, channel_idx=channel_idx, sfreq=sfreq)
    plot_topobutterfly(result, path.parent, sfreq=sfreq)
    plot_topoarray(result, path.parent, sfreq=sfreq)
