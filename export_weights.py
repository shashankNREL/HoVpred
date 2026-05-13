import os
os.environ["DGLBACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import dgl
from gnn import GAT_unc
from molgraph import dgl_molgraph_one_molecule

def main():
    num_hidden = 32
    num_layers = 5
    num_heads = 5
    heads = [num_heads] * num_layers + [1]
    max_atoms = 64
    atom_feat_dim = 16
    equation = ''

    # Initialize model with the best_211007 hyperparameters
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

    # Run a mock forward pass to initialize the weight variables
    device = "/cpu:0"
    g = dgl_molgraph_one_molecule('C', max_atoms, device, False)
    features = g.ndata['feat']
    segment = tf.constant([0] * max_atoms)
    T = tf.constant([298.15], dtype=tf.float32)
    num_mols = tf.constant([1], dtype=tf.int32)

    _ = model(features, g, segment, max_atoms, T, equation, num_mols, training=False)

    # Load weights
    model_path = os.path.join('results_', 'best_211007', 'my_model')
    print(f"Loading weights from {model_path}...")
    model.load_model(model_path)

    # Extract all weights into a dictionary
    weights_dict = {}
    for weight in model.weights:
        # Avoid naming conflicts by replacing slashes
        clean_name = weight.name.replace('/', '_').replace(':', '_')
        weights_dict[clean_name] = weight.numpy()
        print(f"Extracted {weight.name} ({clean_name}): {weight.shape}")

    # Output to a numpy compressed file
    output_path = 'hovpred_weights_best_211007.npz'
    np.savez(output_path, **weights_dict)
    print(f"\nSuccessfully saved {len(weights_dict)} weight tensors to {output_path}")

if __name__ == "__main__":
    main()
