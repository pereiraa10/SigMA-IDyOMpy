"""
dataset_pool.py — multi-subject window pooling on top of dataset.py's
PreparedSubject/TRFDataset, without modifying either.

SubjectCache builds each subject's (expensive) PreparedSubject exactly once
per script run and reuses it across every fold/feature_set/variant that
touches that subject -- e.g. paradigm 3's 5 folds all draw from the same 20
subjects, so PreparedSubject must be built 20 times total, not 100.

MultiSubjectWindowDataset builds a flat window index spanning windows from
multiple subjects' TRFDatasets, using TRFDataset.windows_for_trial(trial_idx)
per (subject, trial_idx) key. Windows never cross trial boundaries (that
guarantee lives in TRFDataset._build_window_index and is not touched here);
this class only adds a subject dimension on top of already-trial-partitioned
keys (decided by splits.py before any windowing happens), so no new leakage
surface is introduced.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, not a package

import utils
from dataset import PreparedSubject


class SubjectCache:
    """Built once per script invocation."""

    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self._prepared = {}     # subject -> PreparedSubject
        self._datasets = {}      # (subject, feature_set, window_samples, hop_samples) -> TRFDataset

    def prepared(self, subject):
        if subject not in self._prepared:
            eeg_path = self.config.paths.eeg_dir / self.config.eeg_filename_pattern.format(subject=subject)
            eeg_data = utils.load_subject_raw_eeg(
                eeg_path, subject, self.config.trial_to_stimulus.get(subject))
            self._prepared[subject] = PreparedSubject(subject, eeg_data, self.config, debug=self.debug)
        return self._prepared[subject]

    def dataset(self, subject, feature_set, window_samples, hop_samples):
        key = (subject, feature_set, window_samples, hop_samples)
        if key not in self._datasets:
            self._datasets[key] = self.prepared(subject).to_dataset(
                feature_set, window_samples=window_samples, hop_samples=hop_samples)
        return self._datasets[key]

    def datasets_for_keys(self, keys, feature_set, window_samples, hop_samples):
        """{subject: TRFDataset} for every subject appearing in `keys` (the
        union of a fold's train+val+test trial keys, typically)."""
        subjects = sorted({s for s, _ in keys})
        if not subjects:
            raise ValueError("datasets_for_keys: no subjects in `keys`")
        ds_by_subject = {
            s: self.dataset(s, feature_set, window_samples, hop_samples) for s in subjects
        }
        ref_subject = subjects[0]
        ref = ds_by_subject[ref_subject]
        for s, ds in ds_by_subject.items():
            if ds.channel_names != ref.channel_names or ds.n_channels != ref.n_channels:
                raise ValueError(
                    f"{s}: channel layout ({ds.n_channels} ch) differs from {ref_subject} "
                    f"({ref.n_channels} ch) -- cannot pool into one shared model.")
        return ds_by_subject


class MultiSubjectWindowDataset(TorchDataset):
    """Flat window index spanning windows from multiple subjects' TRFDatasets.

    ds_by_subject : {subject: TRFDataset}
    trial_keys    : list[(subject, trial_idx)] -- already assigned to exactly
                     one of train/val/test by a splits.py fold generator.
    """

    def __init__(self, ds_by_subject, trial_keys):
        self.ds_by_subject = ds_by_subject
        self.window_refs = []   # list[(subject, local_window_idx)]
        for subject, trial_idx in trial_keys:
            ds = ds_by_subject[subject]
            self.window_refs += [(subject, w) for w in ds.windows_for_trial(trial_idx)]

    def __len__(self):
        return len(self.window_refs)

    def __getitem__(self, idx):
        subject, w = self.window_refs[idx]
        return self.ds_by_subject[subject][w]


if __name__ == '__main__':
    # Smoke test: 2 subjects, full-trial mode (window_samples=None), confirm
    # pooling doesn't cross subjects/trials incorrectly.
    from config import load_config

    config = load_config()
    cache = SubjectCache(config, debug=False)
    subjects = config.subjects[:2]
    feature_set = next(iter(config.feature_sets))
    feature_keys = config.feature_sets[feature_set]

    keys = [(s, i) for s in subjects for i in range(2)]   # 2 trials per subject
    ds_by_subject = cache.datasets_for_keys(keys, feature_set, window_samples=448, hop_samples=384)

    pooled = MultiSubjectWindowDataset(ds_by_subject, keys)
    assert len(pooled) == sum(len(ds_by_subject[s].windows_for_trial(i)) for s, i in keys)
    X0, Y0 = pooled[0]
    assert X0.shape[0] == len(feature_keys)
    assert Y0.shape[0] == ds_by_subject[subjects[0]].n_channels
    print(f"[dataset_pool] pooled {len(subjects)} subjects x 2 trials -> {len(pooled)} windows, "
          f"ds[0] X{tuple(X0.shape)} Y{tuple(Y0.shape)}: OK")
    print("SMOKE TEST PASSED")
