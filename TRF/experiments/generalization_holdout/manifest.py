"""
manifest.py — pure-metadata trial registry for the generalization_holdout
experiments. No EEG is loaded here; fold membership (see splits.py) must be
fully decided from this manifest before any subject's data ever touches disk
or gets windowed, so the "define the experimental unit before creating
overlapping windows" guardrail is structural rather than convention.

Each subject has 30 trials = 10 songs x 3 repeated blocks. For subject S,
`prepared.trials[i]` (built by dataset.PreparedSubject/TRFDataset, 0-indexed,
i in 0..29) always corresponds to event marker `i+1`, and
`song_id = config.trial_to_song_id[i+1]` (1..10); repeat/block index is
`i // 10` (0, 1, or 2). This is guaranteed by construction in utils.py
(markers assigned in loop order in create_mne_raw_from_preprocessed,
mne.find_events returns them sorted by sample, build_trials iterates
range(n_trials) with no reordering/filtering) -- see utils.song_id_for_marker.

subject_type note: config.yaml's subject_type table has Sub1-Sub10 =
Non-musician, Sub11-Sub20 = Musician. Every function here reads
config.subject_type dynamically rather than hardcoding that (or any other)
order.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, not a package

import utils


@dataclass(frozen=True)
class TrialRef:
    subject: str
    trial_idx: int      # 0..29, index into PreparedSubject.to_dataset(...).trials
    song_id: int         # 1..10
    repeat_idx: int        # 0, 1, or 2 (block index)
    subject_type: str       # 'Musician' | 'Non-musician', from config.subject_type

    @property
    def key(self):
        """Hashable identity used everywhere else (Fold.train_keys/val_keys/test_keys,
        SubjectCache/MultiSubjectWindowDataset lookups)."""
        return (self.subject, self.trial_idx)


N_TRIALS_PER_SUBJECT = 30
N_SONGS = 10
N_REPEATS = 3


def build_manifest(config) -> list:
    """One TrialRef per (subject, trial_idx) for every subject in
    config.subjects. Pure metadata -- no eeg_dir access, no PreparedSubject.
    Returns len(config.subjects) * 30 TrialRefs (600 for the full 20-subject
    liberi_dataset config)."""
    refs = []
    for subject in config.subjects:
        subject_type = utils.subject_type_for(subject, config.subject_type)
        for trial_idx in range(N_TRIALS_PER_SUBJECT):
            marker = trial_idx + 1
            song_id = utils.song_id_for_marker(marker, config.trial_to_song_id)
            repeat_idx = trial_idx // N_SONGS
            refs.append(TrialRef(subject, trial_idx, song_id, repeat_idx, subject_type))
    expected = len(config.subjects) * N_TRIALS_PER_SUBJECT
    assert len(refs) == expected, f"build_manifest: got {len(refs)} refs, expected {expected}"
    return refs


def index_by_subject(manifest):
    """{subject: [TrialRef, ...]} -- 30 refs per subject."""
    out = {}
    for ref in manifest:
        out.setdefault(ref.subject, []).append(ref)
    return out


def index_by_subject_song(manifest):
    """{(subject, song_id): [TrialRef, ...]} -- exactly 3 refs (one per repeat) each."""
    out = {}
    for ref in manifest:
        out.setdefault((ref.subject, ref.song_id), []).append(ref)
    return out


def index_by_song(manifest):
    """{song_id: [TrialRef, ...]} -- n_subjects * 3 refs each (60 for 20 subjects)."""
    out = {}
    for ref in manifest:
        out.setdefault(ref.song_id, []).append(ref)
    return out


if __name__ == '__main__':
    from config import load_config

    config = load_config()
    manifest = build_manifest(config)
    print(f"manifest: {len(manifest)} trial refs across {len(config.subjects)} subjects")

    by_subject = index_by_subject(manifest)
    assert all(len(v) == N_TRIALS_PER_SUBJECT for v in by_subject.values())

    by_subject_song = index_by_subject_song(manifest)
    assert all(len(v) == N_REPEATS for v in by_subject_song.values())
    assert len(by_subject_song) == len(config.subjects) * N_SONGS

    by_song = index_by_song(manifest)
    assert all(len(v) == len(config.subjects) * N_REPEATS for v in by_song.values())
    assert len(by_song) == N_SONGS

    n_musician = sum(1 for r in manifest if r.subject_type == 'Musician') // N_TRIALS_PER_SUBJECT
    n_nonmusician = sum(1 for r in manifest if r.subject_type == 'Non-musician') // N_TRIALS_PER_SUBJECT
    print(f"subject_type breakdown: {n_musician} Musician, {n_nonmusician} Non-musician "
          f"(config.yaml ground truth -- do not assume an index-based order)")
    print("SMOKE TEST PASSED")
