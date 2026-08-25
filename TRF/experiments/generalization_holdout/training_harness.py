"""
training_harness.py — generalized SGD training/eval, reusing StimToEEG /
extract_kernel / _pearsonr_channels / _to_tensor directly from TRF_conv.py
(imported, not duplicated). Generalizes TRF_conv.train_one_fold /
loocv_conv from "single subject, leave-one-trial-out" to "an explicit,
possibly-multi-subject train/val/test trial-key split" -- the shape every
splits.py paradigm produces.

Early stopping: unlike train_one_fold (which carves an "inner val trial" out
of its own training trials, since single-subject LOOCV has no other val
set), train_pooled uses the PARADIGM's own val_keys directly to drive early
stopping, because every paradigm here already defines an explicit val set as
part of its design. The metric is the same one train_one_fold uses --
mean, full-trial, channel-averaged Pearson r (_pearsonr_channels) -- just
averaged over however many val trials/subjects are present instead of one.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, not a package

from TRF_conv import (
    StimToEEG, _pearsonr_channels, _to_tensor,
    DEVICE, HIDDEN, N_BLOCKS, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    BATCH_SIZE, SEED,
)

from dataset_pool import MultiSubjectWindowDataset


def train_pooled(ds_by_subject, train_keys, val_keys, n_features, n_channels, variant,
                  epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, patience=EARLY_STOP_PATIENCE,
                  hidden=HIDDEN, n_blocks=N_BLOCKS, batch_size=BATCH_SIZE, seed=SEED,
                  window_dataset_cls=MultiSubjectWindowDataset, debug=False):
    """Pooled multi-subject training. Returns (model, train_mse_history,
    val_mse_history, val_r_history) -- same shape as TRF_conv.train_one_fold.

    window_dataset_cls : the MultiSubjectWindowDataset-shaped class to use
        for building train/val window sets. Overridable so
        null_baseline_mismatched_stimulus.py can pass its
        MismatchedMultiSubjectWindowDataset subclass through this same
        harness without duplicating the training loop.
    """
    torch.manual_seed(seed)
    model = StimToEEG(n_features, n_channels, variant, hidden=hidden, n_blocks=n_blocks).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = window_dataset_cls(ds_by_subject, train_keys)
    val_ds = window_dataset_cls(ds_by_subject, val_keys)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    train_mse_history, val_mse_history, val_r_history = [], [], []
    best_val_r, best_state, since = -np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        epoch_mse = []
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
            epoch_mse.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_mse_batches = []
            for X, Y in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                val_mse_batches.append(loss_fn(model(X), Y).item())

            # Full-trial Pearson r, one term per val trial-key, pooled across
            # however many val subjects there are (early-stopping metric).
            per_trial_r = []
            for subject, trial_idx in val_keys:
                ds = ds_by_subject[subject]
                t = ds.trials[trial_idx]
                Xf = _to_tensor(np.column_stack([t[k] for k in ds.feature_keys]))
                Yf = _to_tensor(t['eeg'])
                per_trial_r.append(_pearsonr_channels(model(Xf), Yf))
            v_r = float(np.mean(per_trial_r))

        train_mse_history.append(float(np.mean(epoch_mse)) if epoch_mse else float('nan'))
        val_mse_history.append(float(np.mean(val_mse_batches)) if val_mse_batches else float('nan'))
        val_r_history.append(v_r)

        if debug:
            print(f"    epoch {epoch}: train_mse={train_mse_history[-1]:.5f} "
                  f"val_mse={val_mse_history[-1]:.5f} val_r={v_r:.4f}")

        if v_r > best_val_r + 1e-6:
            best_val_r = v_r
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_mse_history, val_mse_history, val_r_history


def evaluate_full_trials(model, test_keys_for_one_subject, ds_by_subject, feature_keys):
    """Full-trial inference for one subject's (sorted) test trial keys.
    Returns (Y_pred_concat, Y_true_concat, trial_boundaries) shaped exactly
    as results.build_result expects, using the same offset-accumulation
    pattern as TRF_conv.loocv_conv's held-out eval block."""
    model.eval()
    Y_pred_all, Y_true_all, trial_boundaries = [], [], []
    offset = 0
    for subject, trial_idx in test_keys_for_one_subject:
        ds = ds_by_subject[subject]
        t = ds.trials[trial_idx]
        X = np.column_stack([t[k] for k in feature_keys])
        with torch.no_grad():
            pred = model(_to_tensor(X)).cpu().numpy()[0].T   # (T, n_ch)
        Y_pred_all.append(pred)
        Y_true_all.append(t['eeg'])
        trial_boundaries.append((offset, offset + len(pred)))
        offset += len(pred)
    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all), trial_boundaries
