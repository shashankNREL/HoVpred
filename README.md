# HoVpred

## Installation

The environment requires **Linux with CUDA 11.0**.

```bash
conda env create -f environment.yml
conda activate hovpred
```

## Data

The training data originates from NIST and DIPPR databases, which are not freely available. The CSV files in `data/` contain only the molecular identifiers (SMILES) and temperatures, with the enthalpy of vaporization values redacted. As a result, **only the prediction workflow can be reproduced** using the provided pre-trained model weights; model training and cross-validation require access to the original databases.

## Usage

```
python main.py [-h] [-predict] [-watsoneq] [-K_fold] [-maxatoms MAXATOMS]
               [-lr LR] [-epoch EPOCH] [-batchsize BATCHSIZE] [-layers LAYERS]
               [-heads HEADS] [-residcon] [-explicitH] [-dropout DROPOUT]
               [-modelname MODELNAME] [-num_hidden NUM_HIDDEN] [-train_only]
               [-loss LOSS] [-sw_thr SW_THR] [-sw_decay SW_DECAY]
```

Optional arguments:

```
  -h, --help            show this help message and exit
  -predict              If specified, prediction is carried out
                        (default=False)
  -watsoneq             whether to use watson equation (default=False)
  -K_fold               whether to run KFoldCV (default=False)
  -maxatoms MAXATOMS    Maximum number of atoms in a molecule (default=64)
  -lr LR                Learning rate (default=5.0e-4)
  -epoch EPOCH          epoch (default=200)
  -batchsize BATCHSIZE  batch_size (default=256)
  -layers LAYERS        number of gnn layers (default=5)
  -heads HEADS          number of gat heads (default=5)
  -residcon             whether to use residual connection (default=True)
  -explicitH            whether to use explicit hydrogens (default=False)
  -dropout DROPOUT      dropout rate (default=0.0)
  -modelname MODELNAME  model name (default=an array of hyperparameter values)
  -num_hidden NUM_HIDDEN
                        number of nodes in hidden layers (default=32)
  -train_only           If specified, no 8:1:1 split is carried out, the whole
                        database is used for training (default=False)
  -loss LOSS            loss function (default=mse). Options - mae, mse,
                        kl_div_normal
```

### Prediction

Prepare a `molecules_to_predict.csv` file with two columns: `smiles` and `temperature`, then run:

```bash
python main.py -predict -modelname best_211007 -loss kl_div_normal
```

### Training (requires NIST/DIPPR data)

```bash
python main.py -modelname test_model -loss kl_div_normal
```
non-default hyperparameters can also be tested by adding more arguments