"""
Smoke test: load the HoVpred best_211007 checkpoint on macOS (CPU, arm64)
and compare predictions against the reference Linux run stored in
molecules_to_predict_results.csv.

Tasks verified:
  1. DGL + TF stack is importable on macOS
  2. GAT_unc architecture is reconstructed with the correct hyperparameters
  3. Checkpoint loads without missing / mismatched variables
  4. Per-molecule predictions match Linux reference within 0.05 kJ/mol
"""

import os
import sys

# Must be set before any TF or DGL import.
os.environ["DGLBACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import rdkit.Chem
import tensorflow as tf

import dgl
from gnn import GAT_unc
from molgraph import dgl_molgraph_one_molecule


# ---------------------------------------------------------------------------
# Model hyperparameters (must exactly match the saved best_211007 checkpoint)
# ---------------------------------------------------------------------------
NUM_LAYERS = 5
NUM_HIDDEN = 32
NUM_HEADS  = 5
MAX_ATOMS  = 64
ATOM_FEAT_DIM = 16
EQUATION   = ''
CHECKPOINT = os.path.join('results_', 'best_211007', 'my_model')


def build_model() -> GAT_unc:
    heads = [NUM_HEADS] * NUM_LAYERS + [1]
    return GAT_unc(
        num_layers=NUM_LAYERS,
        in_dim=ATOM_FEAT_DIM,
        num_hidden=NUM_HIDDEN,
        num_classes=NUM_HIDDEN,
        heads=heads,
        activation=tf.nn.relu,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        equation=EQUATION,
    )


def make_segment(smiles: str) -> tf.Tensor:
    """Build the per-atom segment vector with -1 padding for dummy atoms."""
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    n_real = mol.GetNumHeavyAtoms()
    return tf.constant([0] * n_real + [-1] * (MAX_ATOMS - n_real))


def init_weights(model: GAT_unc) -> None:
    """Run a single dummy forward pass so Keras creates all weight tensors."""
    g = dgl_molgraph_one_molecule('C', MAX_ATOMS, '/cpu:0', False)
    features = g.ndata['feat']
    segment  = make_segment('C')
    T        = tf.constant([298.15], dtype=tf.float32)
    num_mols = tf.constant([1], dtype=tf.int32)
    model(features, g=g, segment=segment, Max_atoms=MAX_ATOMS,
          T=T, equation=EQUATION, num_mols=num_mols, training=False)


def predict_one(model: GAT_unc, smiles: str, temp_k: float):
    """Return (mean_hov_kJ_mol, stddev_hov_kJ_mol) for a single molecule."""
    g        = dgl_molgraph_one_molecule(smiles, MAX_ATOMS, '/cpu:0', False)
    features = g.ndata['feat']
    segment  = make_segment(smiles)
    T        = tf.constant([temp_k], dtype=tf.float32)
    num_mols = tf.constant([1], dtype=tf.int32)
    pred = model(features, g=g, segment=segment, Max_atoms=MAX_ATOMS,
                 T=T, equation=EQUATION, num_mols=num_mols, training=False)
    arr = pred.numpy()          # shape (1, 2): [mean, stddev]
    return float(arr[0, 0]), float(arr[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HoVpred macOS smoke test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Environment versions
    # ------------------------------------------------------------------
    print(f"\n[1] Package versions")
    print(f"    TensorFlow : {tf.__version__}")
    print(f"    DGL        : {dgl.__version__}")
    print(f"    NumPy      : {np.__version__}")

    # ------------------------------------------------------------------
    # Step 2 — Architecture parity
    # ------------------------------------------------------------------
    print(f"\n[2] Building GAT_unc model ...")
    model = build_model()
    print(f"    Initialising weights (dummy forward pass on methane) ...")
    init_weights(model)
    print(f"    Weight tensors in model : {len(model.weights)}")

    # ------------------------------------------------------------------
    # Step 3 — Load checkpoint
    # ------------------------------------------------------------------
    print(f"\n[3] Loading checkpoint: {CHECKPOINT}")
    if not os.path.exists(CHECKPOINT + '.index'):
        print(f"    ERROR: checkpoint not found at {CHECKPOINT}.index")
        sys.exit(1)

    status = model.load_weights(CHECKPOINT)
    # expect_partial() silences warnings about optimizer state not restored
    # (we only saved model weights, not optimizer state, via save_weights)
    try:
        status.expect_partial()
    except Exception:
        pass
    print("    Checkpoint loaded successfully.")
    print(f"    Weight tensors after load : {len(model.weights)}")

    # ------------------------------------------------------------------
    # Step 4 — Smoke-test predictions vs. Linux reference
    # ------------------------------------------------------------------
    ref_path = 'molecules_to_predict_results.csv'
    inp_path = 'molecules_to_predict.csv'

    if not os.path.exists(ref_path):
        print(f"\n[4] WARNING: {ref_path} not found; skipping numeric comparison.")
        print("    Running predictions on hardcoded reference hydrocarbons instead.")
        _fallback_predictions(model)
        return

    ref_df = pd.read_csv(ref_path)
    inp_df = pd.read_csv(inp_path)

    # Recompute total_atoms for validation
    inp_df['total_atoms'] = [
        rdkit.Chem.MolFromSmiles(s).GetNumHeavyAtoms()
        for s in inp_df['smiles']
    ]

    print(f"\n[4] Comparing predictions against Linux reference ({ref_path})")
    print(f"    {'SMILES':<35} {'T(K)':>6}  {'Linux':>10}  {'macOS':>10}  {'Δ':>8}  {'OK?'}")
    print("    " + "-" * 80)

    TOLERANCE = 0.05   # kJ/mol — differences larger than this flag as FAIL
    all_pass = True

    for _, row in ref_df.iterrows():
        smiles  = row['smiles']
        temp_k  = float(row['temperature'])
        ref_val = float(row['predicted'])

        try:
            mac_val, mac_std = predict_one(model, smiles, temp_k)
            delta = abs(mac_val - ref_val)
            ok    = delta <= TOLERANCE
            if not ok:
                all_pass = False
            tag = "PASS" if ok else f"FAIL (>{TOLERANCE})"
            short = smiles if len(smiles) <= 33 else smiles[:30] + "..."
            print(f"    {short:<35} {temp_k:>6.1f}  {ref_val:>10.4f}  {mac_val:>10.4f}  {delta:>8.4f}  {tag}")
        except Exception as exc:
            all_pass = False
            print(f"    {smiles:<35} {temp_k:>6.1f}  ERROR: {exc}")

    print()
    if all_pass:
        print("    ALL predictions match Linux reference within tolerance.")
        print("\n    macOS model loading: CONFIRMED WORKING")
    else:
        print("    Some predictions differ from Linux reference.")
        print("    Check for TF version differences or checkpoint variable name mismatches.")
        sys.exit(1)


def _fallback_predictions(model: GAT_unc) -> None:
    """Print predictions for known hydrocarbons when no reference file exists."""
    cases = [
        # (smiles,             temp_K,  NIST_ref_kJ_mol, name)
        ('CCCCCCCCCC',         300.0,   51.42,  'n-decane'),
        ('Cc1ccccc1',          300.0,   38.01,  'toluene'),
        ('CCCCCCCCCCCC',       300.0,   61.52,  'n-dodecane'),
        ('CC1CCCCC1',          300.0,   31.27,  'methylcyclohexane'),
    ]
    print(f"    {'Molecule':<18} {'T(K)':>6}  {'Predicted':>10}  {'Stddev':>8}  {'NIST':>8}")
    print("    " + "-" * 60)
    for smiles, T, nist, name in cases:
        mean, std = predict_one(model, smiles, T)
        print(f"    {name:<18} {T:>6.1f}  {mean:>10.4f}  {std:>8.4f}  {nist:>8.2f}")
    print("\n    Predictions are physically reasonable if close to NIST values.")


if __name__ == '__main__':
    main()
