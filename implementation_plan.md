# Implementation Plan - HoV Comparison (FuelLib vs HoVpred)

This plan outlines the creation of a script to compute Heat of Vaporization (HoV) for all species identified in the `posf10289_init.csv` file using two methods: `FuelLib`'s Group Contribution Method and the `HoVpred` Graph Neural Network model. We will also compute the mixture-averaged HoV for the fuel POSF10289.

## User Review Required

> [!IMPORTANT]
> **Environment Compatibility**: HoVpred was originally trained on Linux with CUDA 11.0 and TensorFlow 2.4.0. Since we are on a macOS environment, loading the raw weights directly might cause compatibility issues. 
> 
> *Resolution:* As suggested, we can export the model weights/parameters on a compatible Linux environment (or a temporary docker container if absolutely necessary) to a platform-agnostic format (e.g., JSON, HDF5, or pure numpy arrays) and then write a small script on the macOS side to parse these weights and build the TensorFlow/DGL model from scratch, or just use the extracted weights to compute the GNN predictions directly. Given that we have access to the model architecture in `gnn.py`, we can reconstruct it on the macOS side with a newer TensorFlow version and load the extracted weights. Alternatively, since it seems like we may struggle to run inference consistently natively with the current packages on Mac (`dgl` is having graphbolt issues on `ct-env`), we could use a lightweight docker container running linux to run the `HoVpred` part of the pipeline and then read the results into the standard script. I will explore extracting weights first or propose using a container if that proves too complex.

## Proposed Changes

### Research & Setup

#### [ACTION] Fix macOS Compatibility for HoVpred
- Resolve the `tensorflow` and `dgl` package issues in the `ct-env` environment. DGL seems to have an issue on Macs relating to `libgraphbolt`. We will either:
    1. Fix the local installation (try removing and reinstalling compatible DGL/TF versions).
    2. Write a script to dump the `best_211007` model weights into standard NumPy/JSON format and rebuild the `GAT_unc` class natively using standard TensorFlow ops without DGL if DGL fails.
    3. If local inference entirely fails, we will suggest running HoVpred in a Linux Docker container and fetching the `molecules_to_predict_results.csv` output.

#### [ACTION] Generate SMILES Mapping
- Parse `posf10289_init.csv` and cross-reference with `FuelLib/fuelData/refCompounds.csv`.
- Create a definitive mapping of `PelePhysics Key` / `Reference Compound` -> `SMILES` for all 67 components in POSF10289. I will generate a small script to query a lightweight database or the `refCompounds.csv` to map these names to SMILES. We MUST have SMILES strings for `HoVpred` to work.

### Prediction Script

#### [NEW] compare_hov.py(file:///Users/syellapa/Documents/Research/2025/SAF/FuelLib/compare_hov.py)
This script will:
1. **Load POSF10289 Data**: Read species and weight fractions from `posf10289_init.csv`.
2. **FuelLib Calculation**:
   - Initialize the `fuel` class for POSF10289.
   - Extract HoV at standard temperature (298 K) for each species (`Hv_stp`).
   - Calculate mixture-averaged HoV using mass fractions.
3. **HoVpred Calculation**:
   - Identify SMILES for all species (mapping PelePhysics keys/Reference Compounds to SMILES).
   - Generate `molecules_to_predict.csv` for HoVpred.
   - Run HoVpred's prediction logic (invoking `main.py` or importing its functions) at 298 K.
   - Extract predictions from `molecules_to_predict_results.csv`.
4. **Comparison & Output**:
   - Create a comparison table/CSV (`hov_comparison_results.csv`).
   - Print mixture-averaged results for both methods.

## Open Questions

- **Temperature**: Should the comparison be done strictly at 298.15 K (STP), or at a specific operating temperature?
- **Linux Execution**: If Mac-native execution of DGL/TF continues to throw dynamic library errors, would you be open to running the HoVpred part in a Docker container or a dedicated Linux environment and passing the results back?

## Verification Plan

### Automated Tests
- Run `compare_hov.py` and check for the existence of `hov_comparison_results.csv`.
- Verify that the mixture-averaged values are printed and physically reasonable (usually ~30-50 kJ/mol for jet fuels).

### Manual Verification
- Inspect the generated CSV to ensure all 67 species have predictions from both methods.
