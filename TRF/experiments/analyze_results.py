"""
analyze_results.py — plotting and aggregation helpers for comparing TRF model
results loaded from the pickles written by results.py (see run_models.ipynb,
part 3 "Examining Results").

Moved out of run_models.ipynb to keep the notebook focused on execution; the
notebook imports these and calls them directly, unchanged from their original
notebook definitions.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statsmodels.stats.multitest import fdrcorrection
from IPython.display import display

import eelbrain
import mne

# Candidate montages to resolve scalp positions from channel *names* alone
# (the pickle schema doesn't save real electrode positions) -- mirrors
# results.py's _CANDIDATE_MONTAGES / _build_sensor fallback list, kept
# separate here since this is an mne Info, not an eelbrain Sensor.
_TOPOMAP_CANDIDATE_MONTAGES = ('biosemi64', 'standard_1020', 'standard_1005')


def permutation_test(a, b, n_permutations=10000, two_tailed=True, seed=42):
    """Sign-flip permutation test for paired data."""
    rng = np.random.default_rng(seed)
    diff = b - a
    observed = diff.mean()

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_mean = (signs * diff).mean()
        if two_tailed:
            count += abs(perm_mean) >= abs(observed)
        else:
            count += perm_mean >= observed

    p_val = count / n_permutations
    return observed, p_val


def plot_model_comparison(models=None, n_subjects=20, pickle_folder='baselines'):
    """
    Create a single figure showing side-by-side violins of mean CV correlation
    (acoustic vs full) for multiple models, followed by a grouped bar chart of
    each subject's mean full-model r, one colored bar per model. Prints a
    p-value / mean-diff table below the plots.
    """
    if models is None:
        models = ['boosting', 'sklearn_ridge', 'mne_ridge', 'conv_nonlinear','conv_linear','conv_separable' ]

    colors = ['steelblue', 'darkorange']
    fig, ax = plt.subplots(figsize=(12, 6))

    spacing = 3
    centers = []
    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='none', label='Acoustic (envelope + onsets)'),
        Patch(facecolor=colors[1], edgecolor='none', label='Full (+ surprisal + entropy)')
    ]

    stats_rows = []
    model_subject_r_full = {}  # model_name -> {sub: mean r across channels, full condition}

    for i, model_name in enumerate(models):
        r_acoustic_all = []
        r_full_all = []
        subs_used = []

        for sub in range(1, n_subjects + 1):
            try:
                a = eelbrain.load.unpickle(f"results/{pickle_folder}/Sub{sub}__{model_name}__acoustic.pkl")
                f = eelbrain.load.unpickle(f"results/{pickle_folder}/Sub{sub}__{model_name}__acoustic_and_surprisal.pkl")
                r_acoustic_all.append(a['r_per_channel'])
                r_full_all.append(f['r_per_channel'])
                subs_used.append(sub)
            except FileNotFoundError:
                # skip missing subjects for this model
                continue

        if len(r_acoustic_all) == 0:
            continue

        r_acoustic_all = np.array(r_acoustic_all)
        r_full_all = np.array(r_full_all)
        r_acoustic_per_sub = r_acoustic_all.mean(axis=1)
        r_full_per_sub = r_full_all.mean(axis=1)
        model_subject_r_full[model_name] = dict(zip(subs_used, r_full_per_sub))

        base = i * spacing
        pos = [base + 1, base + 2]
        centers.append((pos[0] + pos[1]) / 2)

        parts = ax.violinplot([r_acoustic_per_sub, r_full_per_sub],
                              positions=pos, showmedians=False, showextrema=False)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # connect paired subject points within this model
        for j in range(len(r_acoustic_per_sub)):
            ax.plot([pos[0], pos[1]],
                    [r_acoustic_per_sub[j], r_full_per_sub[j]],
                    color='gray', alpha=0.4, linewidth=0.7)

        # scatter with jitter
        jitter = np.random.uniform(-0.04, 0.04, size=len(r_acoustic_per_sub))
        ax.scatter(np.full(len(r_acoustic_per_sub), pos[0]) + jitter, r_acoustic_per_sub,
                   color=colors[0], edgecolors='white', linewidths=0.5, s=40, alpha=0.8, zorder=4)
        jitter = np.random.uniform(-0.04, 0.04, size=len(r_full_per_sub))
        ax.scatter(np.full(len(r_full_per_sub), pos[1]) + jitter, r_full_per_sub,
                   color=colors[1], edgecolors='white', linewidths=0.5, s=40, alpha=0.8, zorder=4)

        # mean ± SEM markers
        acoustic_mean = r_acoustic_per_sub.mean()
        full_mean = r_full_per_sub.mean()
        for k, d in enumerate([r_acoustic_per_sub, r_full_per_sub], start=0):
            mean_val = d.mean()
            sem = d.std() / np.sqrt(len(d))
            ax.errorbar(pos[k], mean_val, yerr=sem, fmt='o', color='black',
                        markersize=6, capsize=4, linewidth=1.5, zorder=5)
            ax.text(pos[k] + 0.08, mean_val, f'Mean: {mean_val:.4f}',
                            color='black', fontsize=10, fontweight='bold', zorder=6) #va='center', ha='left',

        # optional: compute and display p-value for this model
        try:
            mean_diff, p_val = permutation_test(r_acoustic_per_sub, r_full_per_sub,
                                                n_permutations=10000, two_tailed=True)
            sig_label = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            y_bracket = max(r_acoustic_per_sub.max(), r_full_per_sub.max()) + 0.005
            stats_rows.append({'model': model_name,'acoustic_mean_r': acoustic_mean, 'full_mean_r': full_mean, 'p_val': p_val, 'mean_diff': mean_diff})
            ax.plot([pos[0], pos[0], pos[1], pos[1]],
                    [y_bracket - 0.003, y_bracket, y_bracket, y_bracket - 0.003],
                    color='black', linewidth=1.0)
            ax.text((pos[0] + pos[1]) / 2, y_bracket + 0.003, f'p={p_val:.3f} {sig_label}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        except Exception:
            pass

    ax.set_xticks(centers)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_xlim(-0.5, spacing * len(models) - 0.5)
    ax.set_ylabel('Mean CV Correlation (r)', fontsize=12)
    ax.set_title(f'Comparison across models', fontsize=13)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.legend(handles=legend_handles, loc='upper left')
    plt.tight_layout()
    plt.show()

    stats_df = pd.DataFrame(stats_rows)
    display(stats_df)

    # ── Figure 2 – per-subject mean r (full model), grouped bar per model ────────
    # Fixed-order, colorblind-validated categorical palette (one hue per model).
    model_palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                      '#e87ba4', '#008300', '#4a3aa7', '#e34948']

    plotted_models = list(model_subject_r_full.keys())
    all_subs = sorted(set().union(*[set(d) for d in model_subject_r_full.values()])) \
        if plotted_models else []

    if all_subs:
        n_models = len(plotted_models)
        bar_width = 0.8 / n_models
        x = np.arange(len(all_subs))

        fig2, ax2 = plt.subplots(figsize=(max(12, 0.55 * len(all_subs)), 6))
        for m_idx, model_name in enumerate(plotted_models):
            sub_r = model_subject_r_full[model_name]
            heights = [sub_r.get(sub, np.nan) for sub in all_subs]
            offset = (m_idx - (n_models - 1) / 2) * bar_width
            ax2.bar(x + offset, heights, width=bar_width,
                    color=model_palette[m_idx % len(model_palette)],
                    label=model_name, edgecolor='white', linewidth=0.5)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Sub{sub:02d}' for sub in all_subs], rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Mean r across channels (full model)', fontsize=12)
        ax2.set_title('Per-Subject Correlation by Model', fontsize=13)
        ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        plt.show()
    return stats_df


def plot_model_results(model, n_subjects=20, pickle_folder='baselines'):
    """
    Create a violin plot of the mean CV correlation for the specified model and
    evaluate the statistical significance of adding surprisal features (acoustic vs full).

    Additionally, plot the per-subject mean r (averaged across channels)
    """

    # ── 1. Load all subjects ──────────────────────────────────────────────────────
    r_acoustic_all = []  # will be shape (n_subjects, n_channels)
    r_full_all     = []

    print("="*64)
    print(f'Loading {model} Results')
    print("="*64)
    for sub in range(1, n_subjects + 1):
        try:
            data_acoustic = eelbrain.load.unpickle(f"results/{pickle_folder}/Sub{sub}__{model}__acoustic.pkl")
            data_full     = eelbrain.load.unpickle(f"results/{pickle_folder}/Sub{sub}__{model}__acoustic_and_surprisal.pkl")

            r_acoustic_all.append(data_acoustic['r_per_channel'])
            r_full_all.append(data_full['r_per_channel'])
            print(f"  Sub{sub:02d} loaded")

        except FileNotFoundError:
            print(f"  Sub{sub:02d} missing — skipping")

    r_acoustic_all = np.array(r_acoustic_all)  # (n_subjects, n_channels)
    r_full_all     = np.array(r_full_all)
    n_subjects     = len(r_acoustic_all)       # update in case any were skipped

    print(f"\nLoaded {n_subjects} subjects, {r_acoustic_all.shape[1]} channels each")

    # ── 2. Per-subject mean r (averaged across channels) ─────────────────────────
    r_acoustic_per_sub = r_acoustic_all.mean(axis=1)  # (n_subjects,)
    r_full_per_sub     = r_full_all.mean(axis=1)
    r_diff_per_sub     = r_full_per_sub - r_acoustic_per_sub

    # ── 3. Group-level statistics — two-tailed sign-flip permutation test ─────────
    mean_diff, p_val = permutation_test(r_acoustic_per_sub, r_full_per_sub,
                                        n_permutations=10000, two_tailed=True)
    sem_diff = r_diff_per_sub.std() / np.sqrt(n_subjects)

    # Per-channel permutation test across subjects + FDR correction
    n_channels    = r_acoustic_all.shape[1]
    p_per_channel = np.array([
        permutation_test(r_acoustic_all[:, ch], r_full_all[:, ch],
                        n_permutations=10000, two_tailed=True)[1]
        for ch in range(n_channels)
    ])
    _, p_fdr = fdrcorrection(p_per_channel)

    sig_label = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"\nGroup-level permutation test (mean r across channels):")
    print(f"  p = {p_val:.4f}  {sig_label}")
    print(f"  Mean Δr = {mean_diff:.4f} ± {sem_diff:.4f} SEM")
    print(f"  Channels significant after FDR (p<0.05): {(p_fdr < 0.05).sum()} / {n_channels}")


    # ── 4. Figure 1 – Violin: group-level acoustic vs full ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    data   = [r_acoustic_per_sub, r_full_per_sub]
    labels = ['Acoustic\n(envelope + onsets )', 'Full\n(+ surprisal + entropy)']
    colors = ['steelblue', 'darkorange']

    parts = ax.violinplot(data, positions=[1, 2], showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Individual subject points connected by lines
    for i in range(n_subjects):
        ax.plot([1, 2], [r_acoustic_per_sub[i], r_full_per_sub[i]],
                color='gray', alpha=0.4, linewidth=0.8)
        # ax.text(1-0.08,r_acoustic_per_sub[i], f'{r_acoustic_per_sub[i]:.4f}',
        #         fontsize=8, va='center', ha='right', alpha=0.7)
        # ax.text(2 + 0.08, r_full_per_sub[i], f'{r_full_per_sub[i]:.4f}',
        #         fontsize=8, va='center', ha='left', alpha=0.7)

    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.uniform(-0.04, 0.04, size=len(d))
        ax.scatter(i + jitter, d, color=color, s=40, alpha=0.8,
                edgecolors='white', linewidths=0.5, zorder=4)

    # Mean ± SEM
    for i, d in enumerate(data, start=1):
        mean_val = d.mean()

        ax.errorbar(i, mean_val, yerr=d.std() / np.sqrt(len(d)),
                    fmt='o', color='black', markersize=7, capsize=5, linewidth=2, zorder=5)
        ax.text(i + 0.08, mean_val, f'Mean: {mean_val:.4f}',
                color='black', fontsize=10, fontweight='bold', va='center', ha='left', zorder=6)

    # Significance bracket
    y_bracket = max(r_acoustic_per_sub.max(), r_full_per_sub.max()) + 0.005
    ax.plot([1, 1, 2, 2], [y_bracket - 0.003, y_bracket, y_bracket, y_bracket - 0.003],
            color='black', linewidth=1.2)
    ax.text(1.5, y_bracket + 0.003, f'p = {p_val:.4f}  {sig_label}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean CV Correlation (r)', fontsize=12)
    ax.set_title(f'{model} Encoding Correlation\n(N = {n_subjects} subjects)', fontsize=13)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlim(0.4, 2.6)
    plt.tight_layout()
    # plt.savefig('group_violin.png', dpi=150)
    plt.show()

    # ── 5. Figure 2 – Per-subject Δr bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    sub_labels = [f'Sub{i+1:02d}' for i in range(n_subjects)]
    colors_bar = ['seagreen' if d > 0 else 'crimson' for d in r_diff_per_sub]

    ax.bar(range(n_subjects), r_diff_per_sub, color=colors_bar, alpha=0.8, edgecolor='white')
    ax.axhline(0,         color='gray',  linewidth=0.8)
    ax.axhline(mean_diff, color='black', linewidth=1.5, linestyle='--',
            label=f'Group mean Δr = {mean_diff:.4f}')
    ax.set_xticks(range(n_subjects))
    ax.set_xticklabels(sub_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Δr (full − acoustic)')
    ax.set_title(f'{model} Per-Subject Improvement from Adding Surprisal + Entropy\n(green = improvement, red = worse)')
    ax.legend()
    plt.tight_layout()
    # plt.savefig('group_delta_r_per_subject.png', dpi=150)
    plt.show()

    # ── 6. Figure 3 – Per-channel group Δr with FDR-corrected significance ───────
    r_diff_per_channel = (r_full_all - r_acoustic_all).mean(axis=0)  # (n_channels,)
    channel_names      = data_full['meta']['channel_names']

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    # Top: mean Δr per channel
    colors_ch = ['seagreen' if d > 0 else 'crimson' for d in r_diff_per_channel]
    axes[0].bar(range(n_channels), r_diff_per_channel, color=colors_ch, alpha=0.8, edgecolor='white')
    axes[0].axhline(0, color='gray', linewidth=0.8)
    axes[0].axhline(r_diff_per_channel.mean(), color='black', linestyle='--', linewidth=1.2,
                    label=f'Mean = {r_diff_per_channel.mean():.4f}')
    axes[0].set_ylabel('Mean Δr across subjects')
    axes[0].set_title(f'{model} Per-Channel Group Δr (full − acoustic)')
    axes[0].legend()

    # Bottom: -log10(p_fdr) per channel with significance threshold line
    neg_log_p_fdr = -np.log10(np.maximum(p_fdr, 1e-10))
    axes[1].bar(range(n_channels), neg_log_p_fdr, color='mediumpurple', alpha=0.8, edgecolor='white')
    axes[1].axhline(-np.log10(0.05), color='red',    linestyle='--', linewidth=1.2, label='FDR p = 0.05')
    axes[1].axhline(-np.log10(0.01), color='orange', linestyle='--', linewidth=1.2, label='FDR p = 0.01')
    axes[1].set_ylabel('−log₁₀(p_FDR)')
    axes[1].set_title(f'{model} Per-Channel Significance of Improvement (FDR-corrected permutation test)')
    axes[1].set_xticks(range(n_channels))
    axes[1].set_xticklabels(channel_names, rotation=90, fontsize=7)
    axes[1].legend()

    plt.tight_layout()
    # plt.savefig('group_per_channel.png', dpi=150)
    plt.show()

    # ── 7. Figure 4 – Per-subject correlation, double bar by feature set ─────────
    bar_width = 0.35
    x_sub = np.arange(n_subjects)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_sub - bar_width / 2, r_acoustic_per_sub, width=bar_width,
           color=colors[0], label=labels[0].replace('\n', ' '), edgecolor='white', linewidth=0.5)
    ax.bar(x_sub + bar_width / 2, r_full_per_sub, width=bar_width,
           color=colors[1], label=labels[1].replace('\n', ' '), edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x_sub)
    ax.set_xticklabels(sub_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean CV Correlation (r)', fontsize=12)
    ax.set_title(f'{model} Correlation by Subject and Feature Set', fontsize=13)
    ax.legend()
    plt.tight_layout()
    # plt.savefig('per_subject_by_feature_set.png', dpi=150)
    plt.show()

    # ── 8. Figure 5 – Per-channel correlation, double bar by feature set ─────────
    r_acoustic_per_channel = r_acoustic_all.mean(axis=0)  # (n_channels,)
    r_full_per_channel     = r_full_all.mean(axis=0)
    x_ch = np.arange(n_channels)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x_ch - bar_width / 2, r_acoustic_per_channel, width=bar_width,
           color=colors[0], label=labels[0].replace('\n', ' '), edgecolor='white', linewidth=0.5)
    ax.bar(x_ch + bar_width / 2, r_full_per_channel, width=bar_width,
           color=colors[1], label=labels[1].replace('\n', ' '), edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x_ch)
    ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
    ax.set_ylabel('Mean CV Correlation (r)', fontsize=12)
    ax.set_title(f'{model} Correlation by Channel and Feature Set', fontsize=13)
    ax.legend()
    plt.tight_layout()
    # plt.savefig('per_channel_by_feature_set.png', dpi=150)
    plt.show()


def combine_model_results(models=None, n_subjects=20, pickle_folder='baselines'):
    """
    Aggregate pickle results into a long-format pandas DataFrame with one row
    per subject / model / feature_set / trial / channel combination, so all
    higher-level stats (mean_r, per-trial r, etc.) can be computed afterwards
    via groupby on this single granular table.

    Parameters
    - models: list or None -> filter to these model names (e.g. ['sklearn_ridge'])
    - n_subjects: int -> maximum subject index to include (1..n_subjects)
    - pickle_folder: str -> folder inside 'results' to search (default 'baselines')

    Returns
    - pd.DataFrame with columns:
        ['subject', 'subject_idx', 'model', 'feature_set', 'trial',
         'channel', 'channel_idx', 'r', 'file_path']
    Files whose pickle has no 'per_trial_r' (e.g. eelbrain.boosting results,
    which only save the whole-session r_per_channel, see results.py) have no
    trial-level r to explode and are skipped with a warning.
    Also saves a combined pickle at results/{pickle_folder}/combined_results.pkl
    """

    results_dir = Path("results") / pickle_folder
    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} does not exist")

    pattern = re.compile(r'^(Sub\d+)__(.+?)__(.+?)\.pkl$')
    frames = []

    for pkl_path in sorted(results_dir.rglob("*.pkl")):
        name = pkl_path.name
        m = pattern.match(name)
        if not m:
            # skip files not matching expected pattern
            continue
        subject_str, model_name, feature_set = m.groups()
        try:
            subj_idx = int(re.search(r'\d+', subject_str).group())
        except Exception:
            subj_idx = None

        if models is not None and model_name not in models:
            continue
        if n_subjects is not None and subj_idx is not None and subj_idx > n_subjects:
            continue

        try:
            data = eelbrain.load.unpickle(str(pkl_path))
        except Exception:
            # skip unreadable files
            continue

        per_trial_r = data.get('per_trial_r', None)
        if per_trial_r is None:
            print(f"  [warn] {name}: no per_trial_r (whole-session-only result) — skipping")
            continue
        per_trial_r = np.asarray(per_trial_r)  # (n_trials, n_channels)

        meta = data.get('meta', None) or {}
        channel_names = meta.get('channel_names')
        n_trials, n_channels = per_trial_r.shape
        if channel_names is None or len(channel_names) != n_channels:
            channel_names = list(range(n_channels))
        channel_names = np.asarray(channel_names)

        trial_idx, chan_idx = np.meshgrid(
            np.arange(n_trials), np.arange(n_channels), indexing='ij')

        file_df = pd.DataFrame({
            'subject': subject_str,
            'subject_idx': subj_idx,
            'model': model_name,
            'feature_set': feature_set,
            'trial': trial_idx.ravel(),
            'channel': channel_names[chan_idx.ravel()],
            'channel_idx': chan_idx.ravel(),
            'r': per_trial_r.ravel(),
            'file_path': str(pkl_path),
        })
        frames.append(file_df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        'subject', 'subject_idx', 'model', 'feature_set', 'trial',
        'channel', 'channel_idx', 'r', 'file_path'])

    if 'subject_idx' in df.columns:
        df['subject_idx'] = pd.to_numeric(df['subject_idx'], errors='coerce').astype('Int64')

    # save for later reuse
    out_path = results_dir / "combined_results.pkl"
    try:
        df.to_pickle(out_path)
    except Exception:
        pass

    return df


def verify_combine_model_results(combined_df, n_check=10, atol=0.05, seed=42):
    """
    Sanity check for combine_model_results: reloads a random sample of the
    underlying pickle files directly and compares them against combined_df.

      (a) per-channel: mean of combined_df['r'] over trials, per channel,
          vs. that pickle's saved r_per_channel.
      (b) overall: mean of combined_df['r'] over every trial/channel row,
          vs. that pickle's mean_r (r_per_channel.mean()).

    Note this is a closeness check, not an exact-equality one: r_per_channel
    is Pearson r on the whole-session CONCATENATED Y_true/Y_pred (see
    results.py), while combined_df['r'] comes from per_trial_r, computed
    trial-by-trial. Averaging per-trial correlations is not mathematically
    identical to correlating the concatenation, so small differences are
    expected -- this checks they land in the same ballpark, not bit-for-bit.
    """
    rng = np.random.default_rng(seed)
    file_paths = combined_df['file_path'].unique()
    sample = rng.choice(file_paths, size=min(n_check, len(file_paths)), replace=False)

    rows = []
    for fp in sample:
        data = eelbrain.load.unpickle(fp)
        r_per_channel = np.asarray(data['r_per_channel'])
        pickle_mean_r = float(r_per_channel.mean())

        sub_df = combined_df[combined_df['file_path'] == fp]
        per_channel_mean = sub_df.groupby('channel_idx')['r'].mean().sort_index()

        if len(per_channel_mean) != len(r_per_channel):
            print(f"[MISMATCH] {fp}: {len(per_channel_mean)} channels in "
                  f"combined_df vs {len(r_per_channel)} in pickle -- skipping")
            continue

        channel_abs_diff = np.abs(per_channel_mean.to_numpy() - r_per_channel)
        combined_mean_r = sub_df['r'].mean()

        rows.append({
            'file_path': fp,
            'n_channels': len(r_per_channel),
            'n_trials': sub_df['trial'].nunique(),
            'max_channel_abs_diff': channel_abs_diff.max(),
            'mean_channel_abs_diff': channel_abs_diff.mean(),
            'pickle_mean_r': pickle_mean_r,
            'combined_mean_r': combined_mean_r,
            'mean_r_abs_diff': abs(combined_mean_r - pickle_mean_r),
        })

    report = pd.DataFrame(rows)
    with pd.option_context('display.max_colwidth', 40):
        print(report.to_string(index=False))

    within_tol = (report['mean_channel_abs_diff'] < atol) & (report['mean_r_abs_diff'] < atol)
    print()
    if within_tol.all():
        print(f"All {len(report)} sampled files within tolerance (atol={atol}).")
    else:
        print(f"{(~within_tol).sum()} / {len(report)} sampled files EXCEED tolerance (atol={atol}):")
        print(report.loc[~within_tol, 'file_path'].to_string(index=False))

    return report


def _resolve_montage_info(channel_names):
    """
    mne.Info with scalp positions for `channel_names`, resolved by trying
    each of _TOPOMAP_CANDIDATE_MONTAGES in turn (the pickle schema doesn't
    save real electrode positions -- see results._build_sensor, the
    eelbrain-Sensor equivalent of this). Raises if no candidate montage
    covers every channel name.
    """
    for montage_name in _TOPOMAP_CANDIDATE_MONTAGES:
        montage = mne.channels.make_standard_montage(montage_name)
        if all(ch in montage.ch_names for ch in channel_names):
            info = mne.create_info(list(channel_names), sfreq=1.0, ch_types='eeg')
            info.set_montage(montage)
            return info
    raise ValueError(
        f"None of the candidate montages {_TOPOMAP_CANDIDATE_MONTAGES} contain "
        f"every channel name in {list(channel_names)}. Add the correct montage "
        f"to _TOPOMAP_CANDIDATE_MONTAGES in analyze_results.py."
    )


def plot_channel_performance(models, n_subjects=20, pickle_folder='baselines',
                              feature_set='acoustic_and_surprisal', sort_by_r=False,
                              topomap_cmap='viridis'):
    """
    For each model in `models`, plot a bar chart of channel name (x-axis) vs.
    mean Y_pred/Y_true correlation (y-axis), averaged across every subject and
    every trial for that model, plus a scalp topomap of the same per-channel
    means -- one figure per model. Useful for spotting channels that
    consistently predict better (or worse) regardless of subject, e.g.
    auditory electrodes outperforming frontal ones, and whether that pattern
    is spatially coherent (clustered over one region) rather than scattered.

    Uses each pickle's 'per_trial_r' array (n_trials, n_channels), so the
    per-channel mean here is an average of per-trial correlations, same basis
    as combine_model_results/verify_combine_model_results.

    The topomap uses `topomap_cmap` (sequential, default 'viridis') rather
    than the red/white/blue diverging colormap used by the actual-vs-predicted
    ERP topoplots elsewhere in the notebook (plot_topoarray/plot_topobutterfly
    in results.py), so the two aren't visually confused -- those show signed
    signal amplitude, this shows a mean correlation.

    Parameters
    - models: list of model names to plot (e.g. ['sklearn_ridge', 'conv_nonlinear'])
    - n_subjects: max subject index to include (1..n_subjects)
    - pickle_folder: folder inside 'results' to search (default 'baselines')
    - feature_set: which condition to plot, 'acoustic' or 'acoustic_and_surprisal'
    - sort_by_r: if True, order the BAR CHART's bars by descending mean r so
      the strongest channels are easy to read off; if False, keep the
      pickle's channel order. Does not affect the topomap, which is always
      plotted in true scalp position regardless of this setting.
    - topomap_cmap: matplotlib colormap name for the topomap heatmap
    """
    for model in models:
        all_r = []  # list of (n_trials, n_channels) arrays, one per subject
        channel_names = None

        for sub in range(1, n_subjects + 1):
            try:
                data = eelbrain.load.unpickle(
                    f"results/{pickle_folder}/Sub{sub}__{model}__{feature_set}.pkl")
            except FileNotFoundError:
                continue

            per_trial_r = data.get('per_trial_r', None)
            if per_trial_r is None:
                continue  # whole-session-only result (e.g. boosting) -- no trial-level r

            all_r.append(np.asarray(per_trial_r))
            if channel_names is None:
                channel_names = np.asarray(data['meta']['channel_names'])

        if not all_r:
            print(f"[warn] {model}: no per_trial_r data found for feature_set='{feature_set}' -- skipping")
            continue

        stacked = np.concatenate(all_r, axis=0)  # (total_trials_across_subjects, n_channels)
        mean_r = stacked.mean(axis=0)
        sem_r = stacked.std(axis=0) / np.sqrt(stacked.shape[0])

        if sort_by_r:
            order = np.argsort(mean_r)[::-1]
        else:
            order = np.arange(len(mean_r))
        plot_channels = channel_names[order]
        plot_mean = mean_r[order]
        plot_sem = sem_r[order]

        fig, (ax_bar, ax_topo) = plt.subplots(
            1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 1]})

        colors = ['seagreen' if v > 0 else 'crimson' for v in plot_mean]
        ax_bar.bar(range(len(plot_channels)), plot_mean, yerr=plot_sem,
                   color=colors, alpha=0.8, edgecolor='white', linewidth=0.5, capsize=2)
        ax_bar.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax_bar.set_xticks(range(len(plot_channels)))
        ax_bar.set_xticklabels(plot_channels, rotation=90, fontsize=7)
        ax_bar.set_ylabel('Mean r (across subjects & trials)', fontsize=11)
        ax_bar.set_title(
            f'{model} — Per-Channel Correlation ({feature_set}, N={len(all_r)} subjects)',
            fontsize=12)

        # ── Scalp topomap of the same per-channel means, true channel order/
        # position (independent of sort_by_r) ─────────────────────────────
        try:
            info = _resolve_montage_info(channel_names)
            im, _ = mne.viz.plot_topomap(
                mean_r, info, axes=ax_topo, cmap=topomap_cmap,
                vlim=(mean_r.min(), mean_r.max()), sensors=True, contours=6, show=False)
            cbar = plt.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.06)
            cbar.set_label('Mean r', fontsize=9)
            ax_topo.set_title('Scalp map', fontsize=11)
        except ValueError as e:
            ax_topo.axis('off')
            ax_topo.text(0.5, 0.5, f"topomap unavailable:\n{e}",
                         ha='center', va='center', fontsize=8, wrap=True,
                         transform=ax_topo.transAxes)

        plt.tight_layout()
        plt.show()
