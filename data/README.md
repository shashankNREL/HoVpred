# HoV databases

- `Data.csv`: The dataframe with predefined training/validation/test set splits (to do data splits consistent with those in literature and compare the accuracy with literature)
- `Data_for_kfold.csv` for 10-fold cross-validation: `main.py` performs random data split into 10 training/validation folds + one held-out test set

**Note:** The enthalpy of vaporization values originating from NIST and DIPPR have been redacted because these databases are not freely redistributable. The CSV files retain SMILES and temperature columns so that the file structure is preserved, but training and cross-validation cannot be run without populating the missing values from the original sources.
