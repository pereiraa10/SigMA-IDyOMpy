"""
splits.py — the 4 generalization-holdout paradigms, as pure functions over
manifest.py's trial registry. Every generator returns Fold object(s) whose
train_keys/val_keys/test_keys are lists of (subject, trial_idx) -- i.e.
whole TRIALS, decided before any windowing happens. TRFDataset never lets a
window cross a trial boundary (see dataset.py's _build_window_index), so
assigning whole trials to exactly one of train/val/test here is sufficient
to guarantee no window-level leakage downstream.

All 4 paradigms are grouped by subject x stimulus at the trial level, per
the leakage guardrail: no split ever puts different repeats of the same
(subject, song) trial-group into different partitions except paradigm 1,
where that is the entire point (memorization ceiling) and is done at the
whole-repeat (whole-trial) granularity, never at the window level.

Paradigm summary (arithmetic verified against manifest.N_TRIALS_PER_SUBJECT
et al.; see each function's docstring for the derivation):
    1. memorization ceiling      -- 1 split,  420 / 90 / 90   trials (70/15/15%)
    2. within-subj, across-stim  -- 10 folds, 480 / 60 / 60   trials each
    3. across-subj, within-stim  -- 5 folds,  480 / 60 / 60   trials each
    4. across-subj, across-stim  -- 5 folds,  336 / 48 / 24   trials each (408 total, not 600 -- see its docstring)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, not a package

from manifest import (
    N_REPEATS, N_SONGS, N_TRIALS_PER_SUBJECT,
    build_manifest, index_by_song, index_by_subject, index_by_subject_song,
)


@dataclass
class Fold:
    paradigm: str
    fold_idx: int
    train_keys: list
    val_keys: list
    test_keys: list
    train_subjects: list = field(default_factory=list)
    val_subjects: list = field(default_factory=list)
    test_subjects: list = field(default_factory=list)
    train_songs: list = None
    val_songs: list = None
    test_songs: list = None


def assert_fold_integrity(fold, expected_train, expected_val, expected_test):
    """No duplicate keys within a split, pairwise disjointness across splits,
    and exact expected trial counts (the "you should see exactly 42/9/9
    trials, never 40/10/10" guardrail, generalized per-paradigm)."""
    tr, va, te = set(fold.train_keys), set(fold.val_keys), set(fold.test_keys)
    assert len(fold.train_keys) == len(tr), \
        f"fold {fold.fold_idx} ({fold.paradigm}): duplicate key in train_keys"
    assert len(fold.val_keys) == len(va), \
        f"fold {fold.fold_idx} ({fold.paradigm}): duplicate key in val_keys"
    assert len(fold.test_keys) == len(te), \
        f"fold {fold.fold_idx} ({fold.paradigm}): duplicate key in test_keys"
    assert tr.isdisjoint(va), f"fold {fold.fold_idx} ({fold.paradigm}): train/val leakage"
    assert tr.isdisjoint(te), f"fold {fold.fold_idx} ({fold.paradigm}): train/test leakage"
    assert va.isdisjoint(te), f"fold {fold.fold_idx} ({fold.paradigm}): val/test leakage"
    got = (len(tr), len(va), len(te))
    expected = (expected_train, expected_val, expected_test)
    assert got == expected, \
        f"fold {fold.fold_idx} ({fold.paradigm}): got train/val/test counts {got}, expected {expected}"


def log_fold_manifest(fold, out_path=None):
    """Print (and optionally write) the full train/val/test manifest for one
    fold: which subjects/songs are in each split, and the complete sorted
    list of (subject, trial_idx) keys -- for visual leakage auditing."""
    lines = [
        f"=== {fold.paradigm} fold {fold.fold_idx} ===",
        f"train: {len(fold.train_keys)} trials | subjects={fold.train_subjects} | songs={fold.train_songs}",
        f"val:   {len(fold.val_keys)} trials | subjects={fold.val_subjects} | songs={fold.val_songs}",
        f"test:  {len(fold.test_keys)} trials | subjects={fold.test_subjects} | songs={fold.test_songs}",
        "train_keys: " + ", ".join(f"{s}#{i}" for s, i in sorted(fold.train_keys)),
        "val_keys:   " + ", ".join(f"{s}#{i}" for s, i in sorted(fold.val_keys)),
        "test_keys:  " + ", ".join(f"{s}#{i}" for s, i in sorted(fold.test_keys)),
    ]
    text = "\n".join(lines)
    print(text)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    return text


# ════════════════════════════════════════════════════════════════════════════
# Paradigm 1 — within-subject, within-stimulus (memorization ceiling)
# ════════════════════════════════════════════════════════════════════════════

def paradigm1_memorization_ceiling(config, manifest, frac=0.45, seed=0):
    """Single split (not a rotating k-fold -- this paradigm is a sanity/
    ceiling check, not a generalization test).

    Derivation: n_subjects * N_SONGS (subject, song) pairs total. A fraction
    `frac` of them are "split pairs": their 3 repeats go one each to
    train/val/test (repeat0->train, repeat1->val, repeat2->test), keeping
    each whole repeat intact as required. The remaining pairs contribute all
    3 repeats to train only.

    With frac=0.45 and 20 subjects x 10 songs = 200 pairs: 90 split pairs
    (alternating 5-then-4 per subject, seeded, so it sums to exactly 90 over
    20 subjects) + 110 non-split pairs.
        train = 90*1 + 110*3 = 90 + 330 = 420
        val   = 90*1                    = 90
        test  = 90*1                    = 90
        420 + 90 + 90 = 600 total; 70% / 15% / 15% exactly.
    """
    rng = np.random.default_rng(seed)
    by_subject_song = index_by_subject_song(manifest)
    n_subjects = len(config.subjects)
    n_pairs = n_subjects * N_SONGS
    n_split_target = round(frac * n_pairs)

    train_keys, val_keys, test_keys = [], [], []
    n_split_total = 0
    for i, subject in enumerate(config.subjects):
        # Alternate 5-then-4 split-songs per subject so the total sums to
        # n_split_target for the canonical 20-subject/10-song config (10*5 +
        # 10*4 = 90 == round(0.45*200)); for other subject/song counts this
        # still distributes as evenly as possible across subjects.
        remaining_subjects = n_subjects - i
        remaining_target = n_split_target - n_split_total
        n_split_here = min(N_SONGS, round(remaining_target / remaining_subjects))
        n_split_total += n_split_here

        songs = rng.permutation(np.arange(1, N_SONGS + 1))
        split_songs = set(songs[:n_split_here].tolist())
        nonsplit_songs = set(songs[n_split_here:].tolist())

        for song_id in split_songs:
            refs = sorted(by_subject_song[(subject, song_id)], key=lambda r: r.repeat_idx)
            assert len(refs) == N_REPEATS
            train_keys.append(refs[0].key)
            val_keys.append(refs[1].key)
            test_keys.append(refs[2].key)
        for song_id in nonsplit_songs:
            train_keys += [r.key for r in by_subject_song[(subject, song_id)]]

    fold = Fold(
        'paradigm1_memorization_ceiling', 0, train_keys, val_keys, test_keys,
        train_subjects=list(config.subjects), val_subjects=list(config.subjects),
        test_subjects=list(config.subjects),
    )
    n_nonsplit = n_pairs - n_split_total
    expected_train = n_split_total * 1 + n_nonsplit * N_REPEATS
    expected_val = n_split_total
    expected_test = n_split_total
    assert_fold_integrity(fold, expected_train, expected_val, expected_test)
    return fold


# ════════════════════════════════════════════════════════════════════════════
# Paradigm 2 — within-subject, across-stimulus generalization
# ════════════════════════════════════════════════════════════════════════════

def paradigm2_within_subject_across_stimulus(config, manifest):
    """10 folds (one per held-out song rotation), all 20 subjects participate
    identically in every fold -- no subject rotation. Fold k: test_song =
    songs[k], val_song = songs[(k+1) % 10] (always different from test_song),
    train_songs = remaining 8.

    Per fold: train = n_subjects*8*3, val = n_subjects*1*3, test =
    n_subjects*1*3 -- for 20 subjects: 480/60/60, summing to 600.
    """
    by_song = index_by_song(manifest)
    songs = list(range(1, N_SONGS + 1))
    n_subjects = len(config.subjects)

    folds = []
    for k in range(N_SONGS):
        test_song = songs[k]
        val_song = songs[(k + 1) % N_SONGS]
        train_songs = [s for s in songs if s not in (test_song, val_song)]

        train_keys = [r.key for s in train_songs for r in by_song[s]]
        val_keys = [r.key for r in by_song[val_song]]
        test_keys = [r.key for r in by_song[test_song]]

        fold = Fold(
            'paradigm2_within_subject_across_stimulus', k, train_keys, val_keys, test_keys,
            train_subjects=list(config.subjects), val_subjects=list(config.subjects),
            test_subjects=list(config.subjects),
            train_songs=train_songs, val_songs=[val_song], test_songs=[test_song],
        )
        assert_fold_integrity(
            fold, n_subjects * 8 * N_REPEATS, n_subjects * 1 * N_REPEATS, n_subjects * 1 * N_REPEATS)
        folds.append(fold)
    return folds


# ════════════════════════════════════════════════════════════════════════════
# Shared subject-fold machinery for paradigms 3 & 4 (musician-balanced)
# ════════════════════════════════════════════════════════════════════════════

def _stratum_folds(stratum_subjects, n_splits):
    """Run sklearn.model_selection.GroupKFold(n_splits=n_splits) over a
    stratum of subjects (e.g. all 10 musicians), one dummy row per subject,
    groups=subject id. Returns a list of `n_splits` lists of held-out
    subjects (the GroupKFold test-fold membership), each sorted for
    determinism. This is the literal `GroupKFold(groups=subject_id)` the
    leakage guardrail specifies, applied per subject_type stratum (see
    paradigm3/4 docstrings for why plain unstratified GroupKFold over all
    subjects can't guarantee musician/non-musician balance per fold)."""
    subs = sorted(stratum_subjects)
    assert len(subs) % n_splits == 0, \
        f"_stratum_folds: {len(subs)} subjects not evenly divisible by n_splits={n_splits}"
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(subs), 1))
    held_out_per_fold = [None] * n_splits
    for fold_i, (_, test_idx) in enumerate(gkf.split(X_dummy, groups=subs)):
        held_out_per_fold[fold_i] = sorted(subs[i] for i in test_idx)
    return held_out_per_fold


def _musician_strata(config):
    musicians = sorted(s for s in config.subjects if config.subject_type[s] == 'Musician')
    non_musicians = sorted(s for s in config.subjects if config.subject_type[s] == 'Non-musician')
    return musicians, non_musicians


# ════════════════════════════════════════════════════════════════════════════
# Paradigm 3 — across-subject, within-stimulus generalization
# ════════════════════════════════════════════════════════════════════════════

def paradigm3_across_subject_within_stimulus(config, manifest, seed=0):
    """5 folds. GroupKFold(n_splits=5, groups=subject) run separately within
    each subject_type stratum (10 musicians, 10 non-musicians -> 2 held out
    per stratum per fold), zipped by fold index, with each stratum's 2
    held-out subjects split 1->val, 1->test. So every fold's val_subjects =
    [1 musician, 1 non-musician] and test_subjects = [1 musician, 1
    non-musician], guaranteeing balance that plain unstratified GroupKFold
    over all 20 subjects cannot guarantee (its train/test/val fold assembly
    from group sizes has no notion of the musician/non-musician label).

    Per fold: train = 16 subjects * 10 songs * 3 repeats = 480
              val   = 2 subjects * 10 songs * 3 repeats  = 60
              test  = 2 subjects * 10 songs * 3 repeats  = 60
    480+60+60 = 600.
    """
    musicians, non_musicians = _musician_strata(config)
    assert len(musicians) >= 2 and len(non_musicians) >= 2, \
        "paradigm3 needs >=2 subjects per subject_type stratum"
    n_folds = 5
    mus_held = _stratum_folds(musicians, n_folds)
    non_held = _stratum_folds(non_musicians, n_folds)

    by_subject = index_by_subject(manifest)
    folds = []
    for k in range(n_folds):
        mus_pair = mus_held[k]      # exactly 2 musicians
        non_pair = non_held[k]      # exactly 2 non-musicians
        assert len(mus_pair) == 2 and len(non_pair) == 2, \
            f"paradigm3 fold {k}: expected 2 held-out subjects per stratum, " \
            f"got musicians={mus_pair} non_musicians={non_pair}"

        val_subjects = sorted([mus_pair[0], non_pair[0]])
        test_subjects = sorted([mus_pair[1], non_pair[1]])
        heldout = set(val_subjects) | set(test_subjects)
        train_subjects = sorted(s for s in config.subjects if s not in heldout)

        train_keys = [r.key for s in train_subjects for r in by_subject[s]]
        val_keys = [r.key for s in val_subjects for r in by_subject[s]]
        test_keys = [r.key for s in test_subjects for r in by_subject[s]]

        fold = Fold(
            'paradigm3_across_subject_within_stimulus', k, train_keys, val_keys, test_keys,
            train_subjects=train_subjects, val_subjects=val_subjects, test_subjects=test_subjects,
            train_songs=list(range(1, N_SONGS + 1)), val_songs=list(range(1, N_SONGS + 1)),
            test_songs=list(range(1, N_SONGS + 1)),
        )
        assert_fold_integrity(
            fold, len(train_subjects) * N_SONGS * N_REPEATS,
            len(val_subjects) * N_SONGS * N_REPEATS, len(test_subjects) * N_SONGS * N_REPEATS)
        folds.append(fold)

    # Cross-fold coverage check: every subject is held out (as val OR test,
    # never both, never twice) exactly once across the 5 folds -- 5 folds x
    # (2 val + 2 test) = 20 held-out slots for 20 subjects.
    all_heldout = [s for f in folds for s in (f.val_subjects + f.test_subjects)]
    assert sorted(all_heldout) == sorted(config.subjects), \
        "paradigm3: val+test subjects across folds do not partition all subjects exactly once"
    return folds


# ════════════════════════════════════════════════════════════════════════════
# Paradigm 4 — across-subject, across-stimulus generalization (hardest)
# ════════════════════════════════════════════════════════════════════════════

def paradigm4_subject_folds(config):
    """5 folds x 4 held-out test subjects (2 musicians + 2 non-musicians),
    via the same per-stratum GroupKFold mechanism as paradigm 3 -- but here
    only ONE held-out group per fold (no further val/test subject split),
    since paradigm 4's val set reuses the SAME 16 train/val subjects,
    differing only by song (see paradigm4_across_subject_across_stimulus)."""
    musicians, non_musicians = _musician_strata(config)
    n_folds = 5
    mus_held = _stratum_folds(musicians, n_folds)
    non_held = _stratum_folds(non_musicians, n_folds)
    return [sorted(mus_held[k] + non_held[k]) for k in range(n_folds)]  # 5 x 4 subjects


def paradigm4_song_folds():
    """5 folds, deterministic (no RNG needed): 10 songs partitioned into 5
    disjoint test-pairs; fold i's val_song is borrowed from fold (i+1)%5's
    test pair, which is disjoint from fold i's own test pair by
    construction (different partition cell); train_songs = remaining 7.

    Worked example: fold0 test={1,2} val=3 train={4..10}(7)
                     fold1 test={3,4} val=5 train={1,2,6..10}(7)
                     fold2 test={5,6} val=7
                     fold3 test={7,8} val=9
                     fold4 test={9,10} val=1 (wraps to fold0's first test
                     song, disjoint from fold4's own test pair {9,10})
    """
    songs = list(range(1, N_SONGS + 1))
    n_folds = 5
    test_pairs = [tuple(songs[2 * i:2 * i + 2]) for i in range(n_folds)]
    folds = []
    for i in range(n_folds):
        test_songs = list(test_pairs[i])
        val_song = test_pairs[(i + 1) % n_folds][0]
        assert val_song not in test_songs
        train_songs = [s for s in songs if s not in test_songs and s != val_song]
        assert len(train_songs) == 7
        folds.append((train_songs, [val_song], test_songs))
    return folds


def paradigm4_across_subject_across_stimulus(config, manifest):
    """5 folds, zipping a subject-fold with a song-fold.

    Train = 16 subjects x 7 songs x 3 repeats = 336
    Val   = SAME 16 subjects x 1 held-out song x 3 repeats = 48
    Test  = 4 UNSEEN subjects x 2 UNSEEN songs x 3 repeats = 24
    336 + 48 + 24 = 408 -- NOT 600. This is intentional, not a bug: test
    subjects' other 8 songs and train/val subjects' 2 excluded songs
    (val_song of a different fold's rotation + the other test-pair song)
    simply don't fit any of this fold's three cells, so they're excluded
    from the fold entirely rather than forced into an ill-defined role.
    """
    subject_folds = paradigm4_subject_folds(config)
    song_folds = paradigm4_song_folds()
    by_subject_song = index_by_subject_song(manifest)

    folds = []
    for k in range(5):
        test_subjects = subject_folds[k]
        train_val_subjects = sorted(s for s in config.subjects if s not in test_subjects)
        assert len(train_val_subjects) == 16 and len(test_subjects) == 4

        train_songs, val_songs, test_songs = song_folds[k]

        train_keys = [r.key for s in train_val_subjects for song in train_songs
                      for r in by_subject_song[(s, song)]]
        val_keys = [r.key for s in train_val_subjects for song in val_songs
                    for r in by_subject_song[(s, song)]]
        test_keys = [r.key for s in test_subjects for song in test_songs
                     for r in by_subject_song[(s, song)]]

        fold = Fold(
            'paradigm4_across_subject_across_stimulus', k, train_keys, val_keys, test_keys,
            train_subjects=train_val_subjects, val_subjects=train_val_subjects,
            test_subjects=test_subjects, train_songs=train_songs, val_songs=val_songs,
            test_songs=test_songs,
        )
        assert_fold_integrity(
            fold, 16 * 7 * N_REPEATS, 16 * 1 * N_REPEATS, 4 * 2 * N_REPEATS)
        folds.append(fold)

    # Cross-fold coverage: every subject appears as a test subject exactly
    # once (5*4=20, a clean partition); every song appears as a test song
    # exactly once (5*2=10, a clean partition).
    all_test_subjects = [s for f in folds for s in f.test_subjects]
    assert sorted(all_test_subjects) == sorted(config.subjects), \
        "paradigm4: test_subjects across folds do not partition all subjects exactly once"
    all_test_songs = [s for f in folds for s in f.test_songs]
    assert sorted(all_test_songs) == list(range(1, N_SONGS + 1)), \
        "paradigm4: test_songs across folds do not partition all songs exactly once"
    return folds


if __name__ == '__main__':
    from config import load_config

    config = load_config()
    manifest = build_manifest(config)

    fold1 = paradigm1_memorization_ceiling(config, manifest)
    print(f"[paradigm1] train={len(fold1.train_keys)} val={len(fold1.val_keys)} test={len(fold1.test_keys)}")

    folds2 = paradigm2_within_subject_across_stimulus(config, manifest)
    print(f"[paradigm2] {len(folds2)} folds, e.g. fold0: "
          f"train={len(folds2[0].train_keys)} val={len(folds2[0].val_keys)} test={len(folds2[0].test_keys)}")

    folds3 = paradigm3_across_subject_within_stimulus(config, manifest)
    print(f"[paradigm3] {len(folds3)} folds, e.g. fold0: "
          f"train={len(folds3[0].train_keys)} val={len(folds3[0].val_keys)} test={len(folds3[0].test_keys)} "
          f"val_subjects={folds3[0].val_subjects} test_subjects={folds3[0].test_subjects}")

    folds4 = paradigm4_across_subject_across_stimulus(config, manifest)
    print(f"[paradigm4] {len(folds4)} folds, e.g. fold0: "
          f"train={len(folds4[0].train_keys)} val={len(folds4[0].val_keys)} test={len(folds4[0].test_keys)} "
          f"test_subjects={folds4[0].test_subjects} test_songs={folds4[0].test_songs}")

    print("SMOKE TEST PASSED")
