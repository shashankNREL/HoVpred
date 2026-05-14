"""
compare_hov.py — Compare FuelLib GCM vs HoVpred GNN for POSF10289 or POSF10325 at 298.15 K.

Usage
-----
    python compare_hov.py [posf10289|posf10325]   (default: posf10325)

Output
------
    hov_comparison_<fuel>.csv   per-species table with both predictions
    stdout                      summary table + mixture-averaged HoV

Bugs fixed vs. earlier draft
-----------------------------
  Bug 1  DGLBACKEND env var now set before any TF/DGL import.
  Bug 2  Segment tensor uses -1 for dummy (padding) atoms, not 0.
  Bug 3  FuelLib path uses ../FuelLib relative to script location (not cwd).
  Bug 4  Weights loaded directly from TF checkpoint; NPZ export bypassed.
  Bug 5  (moot — no NPZ weight filter needed anymore).
  Bug 6  Isoparaffin SMILES corrected in the source CSV files.
  Bug 7  Mixture average normalised over covered mass fraction only.
  Bug 8  Removed unused fuellib_mix_hv_mol variable.
  Bug 9  FuelLib index built once as a dict, not re-searched per species.
"""

import argparse
import os
import sys

# ── Bug 1: DGLBACKEND must be set before importing tensorflow or dgl ──────────
os.environ["DGLBACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import rdkit.Chem
import tensorflow as tf
import dgl  # noqa: F401 (imported so the backend is initialised)

# ── Bug 3: resolve FuelLib relative to this script, not the caller's cwd ─────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FUELLIB_ROOT = os.path.abspath(os.path.join(_HERE, "../FuelLib"))
sys.path.insert(0, os.path.join(_FUELLIB_ROOT, "source"))
from FuelLib import fuel  # noqa: E402 (must come after sys.path update)

from gnn import GAT_unc
from molgraph import dgl_molgraph_one_molecule

# ── Model constants (must match best_211007 checkpoint exactly) ───────────────
_NUM_LAYERS    = 5
_NUM_HIDDEN    = 32
_NUM_HEADS     = 5
_MAX_ATOMS     = 64
_ATOM_FEAT_DIM = 16
_CHECKPOINT    = os.path.join(_HERE, "results_", "best_211007", "my_model")
TEMP_K         = 298.15


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_segment(smiles: str) -> tf.Tensor:
    """Bug 2 fix: pad with -1 so dummy atoms are excluded from segment mean."""
    mol = rdkit.Chem.MolFromSmiles(smiles)
    n_real = mol.GetNumHeavyAtoms()
    return tf.constant([0] * n_real + [-1] * (_MAX_ATOMS - n_real))


def build_and_load_model() -> GAT_unc:
    """Reconstruct GAT_unc, run dummy forward pass, then load TF checkpoint."""
    heads = [_NUM_HEADS] * _NUM_LAYERS + [1]
    model = GAT_unc(
        num_layers=_NUM_LAYERS,
        in_dim=_ATOM_FEAT_DIM,
        num_hidden=_NUM_HIDDEN,
        num_classes=_NUM_HIDDEN,
        heads=heads,
        activation=tf.nn.relu,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        equation="",
    )
    # Dummy pass on methane to force Keras to create all weight variables.
    g_dummy = dgl_molgraph_one_molecule("C", _MAX_ATOMS, "/cpu:0", False)
    model(
        g_dummy.ndata["feat"],
        g=g_dummy,
        segment=_make_segment("C"),
        Max_atoms=_MAX_ATOMS,
        T=tf.constant([TEMP_K], dtype=tf.float32),
        equation="",
        num_mols=tf.constant([1], dtype=tf.int32),
        training=False,
    )
    # Bug 4 fix: load directly from the TF checkpoint (no NPZ export needed).
    status = model.load_weights(_CHECKPOINT)
    status.expect_partial()
    return model


def predict_hov_batch(model: GAT_unc, smiles_list: list, temp_k: float = TEMP_K) -> list:
    """Return predicted HoV mean (kJ/mol) for each SMILES; NaN for invalid entries."""
    results = []
    for smiles in smiles_list:
        if pd.isna(smiles) or str(smiles).strip() in ("", "NOT_FOUND"):
            results.append(np.nan)
            continue
        mol = rdkit.Chem.MolFromSmiles(str(smiles).strip())
        if mol is None or mol.GetNumHeavyAtoms() > _MAX_ATOMS:
            results.append(np.nan)
            continue
        g = dgl_molgraph_one_molecule(str(smiles).strip(), _MAX_ATOMS, "/cpu:0", False)
        pred = model(
            g.ndata["feat"],
            g=g,
            segment=_make_segment(str(smiles).strip()),  # Bug 2 fix
            Max_atoms=_MAX_ATOMS,
            T=tf.constant([temp_k], dtype=tf.float32),
            equation="",
            num_mols=tf.constant([1], dtype=tf.int32),
            training=False,
        )
        results.append(float(pred.numpy()[0, 0]))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FuelLib helper
# ─────────────────────────────────────────────────────────────────────────────

def fuellib_properties(fuel_obj) -> dict:
    """
    Return per-species FuelLib data keyed by PelePhysics key.

    Values:
        hv_kj_mol  Heat of vaporisation at STP (kJ/mol)
        mw_kg_mol  Molecular weight (kg/mol)
        Y_0        Normalised mass fraction
    """
    # Bug 9 fix: build the lookup dict once.
    return {
        k: {
            "hv_kj_mol": fuel_obj.Hv_stp[i] / 1000.0,
            "mw_kg_mol": fuel_obj.MW[i],
            "Y_0":       fuel_obj.Y_0[i],
        }
        for i, k in enumerate(fuel_obj.pelephysics_keys or [])
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mixture average
# ─────────────────────────────────────────────────────────────────────────────

def mixture_hov_kgkg(
    hov_kj_mol: np.ndarray,
    mw_kg_mol:  np.ndarray,
    mass_frac:  np.ndarray,
) -> tuple[float, float]:
    """
    Mass-weighted mixture HoV in kJ/kg.

    Only species with non-NaN HoV and non-NaN MW are included.
    Returns (mix_hov_kJ_kg, covered_mass_fraction).

    Bug 7 fix: normalise over the covered mass fraction, not the full sum.
    """
    valid   = ~(np.isnan(hov_kj_mol) | np.isnan(mw_kg_mol))
    covered = float(mass_frac[valid].sum())
    if covered == 0.0:
        return np.nan, 0.0
    mix = float(np.sum(hov_kj_mol[valid] / mw_kg_mol[valid] * mass_frac[valid]) / covered)
    return mix, covered


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FuelLib GCM vs HoVpred GNN for a fuel at 298.15 K"
    )
    parser.add_argument(
        "fuel_name",
        nargs="?",
        choices=["posf10289", "posf10325"],
        default="posf10325",
        help="Fuel to analyse (default: posf10325)",
    )
    args   = parser.parse_args()
    fname  = args.fuel_name
    csv_in = os.path.join(_HERE, f"{fname}_with_smiles.csv")

    if not os.path.exists(csv_in):
        sys.exit(f"ERROR: {csv_in} not found. Generate the SMILES CSV first.")

    print(f"\n{'='*65}")
    print(f"  HoV comparison — {fname.upper()} at {TEMP_K} K")
    print(f"{'='*65}")

    # ── Load SMILES data ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_in)

    # ── FuelLib GCM ──────────────────────────────────────────────────────────
    print(f"\n[1] FuelLib GCM  ({fname}) ...")
    fuel_obj  = fuel(fname)
    fl_props  = fuellib_properties(fuel_obj)

    hov_lib, mw_arr, Y_arr = [], [], []
    for key in df["PelePhysics Key"].str.strip():
        props = fl_props.get(key)
        if props:
            hov_lib.append(props["hv_kj_mol"])
            mw_arr.append(props["mw_kg_mol"])
            Y_arr.append(props["Y_0"])
        else:
            hov_lib.append(np.nan)
            mw_arr.append(np.nan)
            Y_arr.append(0.0)

    df["FuelLib_kJmol"] = hov_lib
    df["MW_kgmol"]      = mw_arr
    df["Y_0"]           = Y_arr

    # ── HoVpred GNN ──────────────────────────────────────────────────────────
    print(f"[2] Loading HoVpred checkpoint ...")
    model = build_and_load_model()
    print(f"[3] HoVpred predictions ...")
    df["HoVpred_kJmol"] = predict_hov_batch(model, df["SMILES"].tolist())

    # ── Per-species table ─────────────────────────────────────────────────────
    print(f"\n{'Compound':<24} {'Key':<17} {'Wt%':>5}  {'FuelLib':>9}  {'HoVpred':>9}  {'Δ':>7}")
    print("  " + "-" * 73)
    for _, row in df.iterrows():
        lib   = row["FuelLib_kJmol"]
        pred  = row["HoVpred_kJmol"]
        delta = pred - lib if not (np.isnan(lib) or np.isnan(pred)) else np.nan
        lib_s   = f"{lib:9.2f}"   if not np.isnan(lib)   else f"{'N/A':>9}"
        pred_s  = f"{pred:9.2f}"  if not np.isnan(pred)  else f"{'N/A':>9}"
        delta_s = f"{delta:+7.2f}" if not np.isnan(delta) else f"{'N/A':>7}"
        print(
            f"{str(row['Compound']):<24} {str(row['PelePhysics Key']):<17}"
            f" {row['Weight %']:>5.2f}  {lib_s}  {pred_s}  {delta_s}"
        )

    # ── Mixture averages ──────────────────────────────────────────────────────
    mw  = df["MW_kgmol"].to_numpy()
    Y   = df["Y_0"].to_numpy()
    hov_lib_arr  = df["FuelLib_kJmol"].to_numpy()
    hov_pred_arr = df["HoVpred_kJmol"].to_numpy()

    # Bug 7 fix: each method normalised over its own covered mass fraction.
    mix_lib,  cov_lib  = mixture_hov_kgkg(hov_lib_arr,  mw, Y)
    mix_pred, cov_pred = mixture_hov_kgkg(hov_pred_arr, mw, Y)

    print(f"\n{'─'*65}")
    print(f"  Mixture-averaged HoV at {TEMP_K} K")
    print(f"{'─'*65}")
    print(f"  FuelLib  GCM : {mix_lib:7.2f} kJ/kg  (covers {cov_lib*100:5.1f}% of mass)")
    print(f"  HoVpred  GNN : {mix_pred:7.2f} kJ/kg  (covers {cov_pred*100:5.1f}% of mass)")
    if not (np.isnan(mix_lib) or np.isnan(mix_pred)):
        diff = mix_pred - mix_lib
        pct  = diff / mix_lib * 100
        print(f"  Difference   : {diff:+7.2f} kJ/kg  ({pct:+.1f}%)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_cols = [
        "Compound", "Reference Compound", "PelePhysics Key",
        "Weight %", "SMILES", "FuelLib_kJmol", "HoVpred_kJmol",
    ]
    out_csv = os.path.join(_HERE, f"hov_comparison_{fname}.csv")
    df[out_cols].to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")


if __name__ == "__main__":
    main()
