"""
analyze_holdout_results.py — cross-fold/cross-paradigm result aggregation
for generalization_holdout, mirroring analyze_results.combine_model_results'
shape/columns but adapted for this folder's nested per-paradigm/per-fold
layout (results/<paradigm>/fold<k>/*.pkl instead of a single flat
results/<pickle_folder>/ directory).

Why not reuse analyze_results.combine_model_results directly: it hardcodes
a cwd-relative `Path("results") / pickle_folder` glob (non-recursive) and
its regex only captures (subject, model_tag, feature_set) -- no fold/
paradigm columns, and this folder's pickles live several directories deep.
compare_models.discover_results IS reused as-is (see run scripts /
__main__ below) for single-directory (i.e. single fold) comparisons, since
it already accepts an absolute Path and doesn't assume `Sub\\d+` naming.

permutation_test is reused as-is from compare_models (identical
implementation to analyze_results.permutation_test).
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, not a package

from compare_models import permutation_test  # noqa: F401 (re-exported for callers)

# from results_io import BASE_RESULTS_DIR

_FILENAME_RE = re.compile(r'^(.+?)__(.+?)__(.+?)\.pkl$')


def combine_holdout_results(results_dir=None, models=None):  #changed from BASE_RESULTS_DIR to None so i can run this outside the folder
    """Load every `*__*__*.pkl` under `results_dir` (recursively, so it
    covers every paradigm/fold subdirectory) into a long-format DataFrame:
    one row per (subject, channel), with paradigm/fold/model/feature_set/r
    columns -- the per-channel explode mirrors
    analyze_results.combine_model_results' `channel` row granularity.

    `models`, if given, restricts to those model_tag values (e.g.
    ['conv_linear', 'conv_nonlinear', 'conv_null_nonlinear']).
    """
    rows = []
    for path in sorted(Path(results_dir).rglob('*.pkl')):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        subject_from_name, model_tag, feature_set_from_name = m.groups()

        with open(path, 'rb') as f:
            result = pickle.load(f)
        meta = result['meta']
        model_tag = model_tag if not meta.get('model_variant') else \
            f"{meta['model_family']}_{meta['model_variant']}"
        if models is not None and model_tag not in models:
            continue

        r_per_channel = np.asarray(result['r_per_channel'])
        channel_names = meta.get('channel_names') or list(range(len(r_per_channel)))
        for ch_idx, (ch_name, r) in enumerate(zip(channel_names, r_per_channel)):
            rows.append({
                'subject': meta['subject'],
                'subject_type': meta.get('subject_type'),
                'paradigm': meta.get('paradigm'),
                'fold': meta.get('fold'),
                'model_tag': model_tag,
                'feature_set': meta['feature_set'],
                'channel': ch_name,
                'channel_idx': ch_idx,
                'r': r,
                'null_baseline': meta.get('null_baseline', False),
                'file_path': str(path),
            })
        del result

    if not rows:
        raise ValueError(f"No result pickles found under {results_dir}")
    return pd.DataFrame(rows)


def verify_combine_holdout_results(combined_df, n_check=10, atol=1e-6, seed=42):
    """Spot-check combine_holdout_results against the raw pickles it read
    from, mirroring analyze_results.verify_combine_model_results."""
    rng = np.random.default_rng(seed)
    paths = combined_df['file_path'].unique()
    sample_paths = rng.choice(paths, size=min(n_check, len(paths)), replace=False)
    for path in sample_paths:
        with open(path, 'rb') as f:
            result = pickle.load(f)
        df_rows = combined_df[combined_df['file_path'] == path].sort_values('channel_idx')
        expected_r = np.asarray(result['r_per_channel'])
        got_r = df_rows['r'].to_numpy()
        assert np.allclose(got_r, expected_r, atol=atol), f"mismatch for {path}"
    print(f"verify_combine_holdout_results: {len(sample_paths)} pickles checked, all match")


if __name__ == '__main__':
    df = combine_holdout_results()
    print(f"Loaded {len(df)} rows from {df['file_path'].nunique()} pickles under {BASE_RESULTS_DIR}")
    print(df.groupby(['paradigm', 'model_tag', 'feature_set'])['r']
          .agg(['mean', 'std', 'count']).round(4).to_string())
    verify_combine_holdout_results(df)
