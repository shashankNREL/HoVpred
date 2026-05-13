import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# Force DGL to use TensorFlow backend
os.environ["DGLBACKEND"] = "tensorflow"
import dgl

# Add FuelLib to path
FUELLIB_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../../FuelLib"))
sys.path.append(os.path.join(FUELLIB_ROOT, "source"))

from FuelLib import fuel
from gnn import GAT_unc
from molgraph import dgl_molgraph_one_molecule

def load_native_model(weights_path):
    """Loads the GAT_unc architecture and populates it with weights from .npz file."""
    num_hidden = 32
    num_layers = 5
    num_heads = 5
    heads = [num_heads] * num_layers + [1]
    max_atoms = 64
    atom_feat_dim = 16
    equation = ''

    model = GAT_unc(num_layers=num_layers,
                    in_dim=atom_feat_dim,
                    num_hidden=num_hidden,
                    num_classes=num_hidden,
                    heads=heads,
                    activation=tf.nn.relu,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    negative_slope=0.2,
                    residual=True,
                    equation=equation)

    # Initialize weights with a dummy pass
    g = dgl_molgraph_one_molecule('C', max_atoms, "/cpu:0", False)
    features = g.ndata['feat']
    segment = tf.constant([0] * max_atoms)
    T = tf.constant([298.15], dtype=tf.float32)
    num_mols = tf.constant([1], dtype=tf.int32)
    _ = model(features, g=g, segment=segment, Max_atoms=max_atoms, T=T, 
              equation=equation, num_mols=num_mols, training=False)

    # Load shared weights from NPZ
    print(f"Loading weights from {weights_path}...")
    weights_npz = np.load(weights_path)
    
    # Keras adds indices to layer names (e.g. 'dense_1'). 
    # We will try to map by matching the components (gat_conv vs dense vs T_embedding) 
    # and their relative order.
    
    # Get all weight names from the NPZ
    npz_keys = list(weights_npz.keys())
    
    # Filter keys by categories
    gat_keys = sorted([k for k in npz_keys if 'gat_conv' in k])
    dense_keys = sorted([k for k in npz_keys if 'dense' in k and 'gat_conv' not in k and 'T_embedding' not in k])
    t_embed_keys = sorted([k for k in npz_keys if 'T_embedding' in k])
    readout_keys = sorted([k for k in npz_keys if 'readout' in k])

    print(f"Found {len(gat_keys)} GAT weights, {len(dense_keys)} Dense weights, {len(t_embed_keys)} T-embed weights, {len(readout_keys)} Readout weights.")

    # Sort model weights to ensure consistent assignment
    model_gat_weights = [w for w in model.weights if 'gat_conv' in w.name]
    model_dense_weights = [w for w in model.weights if 'dense' in w.name and 'gat_conv' not in w.name and 'T_embedding' not in w.name and 'readout' not in w.name]
    model_t_embed_weights = [w for w in model.weights if 'T_embedding' in w.name]
    model_readout_weights = [w for w in model.weights if 'readout' in w.name]

    def assign_batch(model_vars, npz_keys):
        for i, v in enumerate(model_vars):
            if i < len(npz_keys):
                v.assign(weights_npz[npz_keys[i]])
                # print(f"Mapped {v.name} <- {npz_keys[i]}")

    assign_batch(model_gat_weights, gat_keys)
    assign_batch(model_dense_weights, dense_keys)
    assign_batch(model_t_embed_weights, t_embed_keys)
    assign_batch(model_readout_weights, readout_keys)
    
    return model

def predict_hov(model, smiles_list, temp_k=298.15):
    """Predict HoV using the loaded model for a list of SMILES."""
    max_atoms = 64
    equation = ''
    results = []
    
    for smiles in smiles_list:
        if smiles == "NOT_FOUND":
            results.append(np.nan)
            continue
            
        try:
            g = dgl_molgraph_one_molecule(smiles, max_atoms, "/cpu:0", False)
            features = g.ndata['feat']
            segment = tf.constant([0] * max_atoms)
            T = tf.constant([temp_k], dtype=tf.float32)
            num_mols = tf.constant([1], dtype=tf.int32)
            
            # Predict
            # GAT_unc returns [mu, sigma] for each data point
            pred = model(features, g=g, segment=segment, Max_atoms=max_atoms, T=T, 
                         equation=equation, num_mols=num_mols, training=False)
            
            # pred is shape (batch_size, 2) where [:, 0] is mu (mean) and [:, 1] is log_var or sigma
            results.append(float(pred.numpy()[0, 0]))
        except Exception as e:
            print(f"Error predicting for {smiles}: {e}")
            results.append(np.nan)
            
    return results

def main():
    # 1. Load weights and initialize HoVpred model
    weights_path = "hovpred_weights_best_211007.npz"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Please ensure the weights are exported and placed here.")
        return
    
    model = load_native_model(weights_path)
    
    # 2. Load the fuel SMILES mapping
    fuel_data_path = "posf10289_with_smiles.csv"
    if not os.path.exists(fuel_data_path):
        print(f"Error: {fuel_data_path} not found. Run the SMILES mapping script first.")
        return
    
    df = pd.read_csv(fuel_data_path)
    smiles_list = df['SMILES'].tolist()
    weights_pct = df['Weight %'].tolist()
    
    # 3. Predict HoV using HoVpred
    print("Running HoVpred predictions...")
    hov_pred = predict_hov(model, smiles_list, temp_k=298.15)
    df['HoVpred [kJ/mol]'] = hov_pred
    
    # 4. Calculate HoV using FuelLib (Group Contribution)
    print("Calculating FuelLib (Group Contribution) HoV...")
    # FuelLib's Hv_stp is at 298.15K usually
    # We initialize a fuel object for the mixture to get component values
    # Note: FuelLib's posf10289 handler might be easier
    
    # Since we want individual species comparison, we'll iterate through PelePhysics keys
    # and use FuelLib's internal data.
    
    # Initialize FuelLib fuel object (for POSF10289)
    
    try:
        my_fuel = fuel('posf10289')
        
        fuellib_hov_mol = []
        for key in df['PelePhysics Key']:
            # Find the component in my_fuel
            idx = -1
            if my_fuel.pelephysics_keys:
                for i, k in enumerate(my_fuel.pelephysics_keys):
                    if k == key:
                        idx = i
                        break
            
            if idx != -1:
                # Hv_stp in FuelLib is J/mol
                hv_j_mol = my_fuel.Hv_stp[idx]
                hv_mol = hv_j_mol / 1000.0 # kJ/mol
                fuellib_hov_mol.append(hv_mol)
            else:
                fuellib_hov_mol.append(np.nan)
        
        df['FuelLib [kJ/mol]'] = fuellib_hov_mol
    except Exception as e:
        print(f"Error calculating FuelLib properties: {e}")
        import traceback
        traceback.print_exc()
        df['FuelLib [kJ/mol]'] = np.nan

    # 5. Calculate Mixture Averages
    
    # Mass-weighted mixture HoV
    # FuelLib Y_0 is mass fraction
    fuellib_mix_hv_mol = 0
    fuellib_mix_hv_kg = 0
    hovpred_mix_hv_kg = 0
    
    for i, row in df.iterrows():
        idx = -1
        if 'my_fuel' in locals() and my_fuel.pelephysics_keys:
            for j, k in enumerate(my_fuel.pelephysics_keys):
                if k == row['PelePhysics Key']:
                    idx = j
                    break
        
        if idx != -1:
            mw_kg_mol = my_fuel.MW[idx] # kg/mol
            mass_frac = my_fuel.Y_0[idx]
            
            # FuelLib Mixture Avg
            hv_lib_kj_mol = row['FuelLib [kJ/mol]']
            if not np.isnan(hv_lib_kj_mol):
                fuellib_mix_hv_kg += (hv_lib_kj_mol / mw_kg_mol) * mass_frac
            
            # HoVpred Mixture Avg
            hv_pred_kj_mol = row['HoVpred [kJ/mol]']
            if not np.isnan(hv_pred_kj_mol):
                hovpred_mix_hv_kg += (hv_pred_kj_mol / mw_kg_mol) * mass_frac
    
    print(f"\nMixture HoV (Mass-Weighted Average) at 298.15K:")
    print(f"FuelLib (GCM): {fuellib_mix_hv_kg:.2f} kJ/kg")
    print(f"HoVpred (GNN): {hovpred_mix_hv_kg:.2f} kJ/kg")
    if fuellib_mix_hv_kg > 0:
        print(f"Difference: {abs(fuellib_mix_hv_kg - hovpred_mix_hv_kg):.2f} kJ/kg ({(abs(fuellib_mix_hv_kg - hovpred_mix_hv_kg)/fuellib_mix_hv_kg)*100:.2f}%)")

    # 6. Save final comparison
    df.to_csv("hov_comparison_posf10289.csv", index=False)
    print("\nDetailed comparison saved to hov_comparison_posf10289.csv")

if __name__ == "__main__":
    main()
