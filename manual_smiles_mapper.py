import pandas as pd

fuel_csv = 'posf10289_with_smiles.csv'
df = pd.read_csv(fuel_csv)

# Manual dictionary for known posf10289 fuel components
smiles_map = {
    'toluene': 'Cc1ccccc1',
    'ethyl benzene': 'CCc1ccccc1',
    'propyl benzene': 'CCCc1ccccc1',
    'butyl benzene': 'CCCCc1ccccc1',
    'pentyl benzene': 'CCCCCc1ccccc1',
    'hexyl benzene': 'CCCCCCc1ccccc1',
    'heptyl benzene': 'CCCCCCCc1ccccc1',
    'octyl benzene': 'CCCCCCCCc1ccccc1',
    'nonyl benzene': 'CCCCCCCCCc1ccccc1',
    'naphthalene': 'c1ccc2ccccc2c1',
    '1-methyl naphthalene': 'Cc1cccc2ccccc12',
    '1-ethyl naphthalene': 'CCc1cccc2ccccc12',
    '1-propyl naphthalene': 'CCCc1cccc2ccccc12',
    'indane': 'C1CCc2ccccc12',
    'tetralin': 'C1CCC2=CC=CC=C2C1',
    '2-methyl tetralin': 'CC1CC2=CC=CC=C2CC1',
    '2-ethly tetralin': 'CCC1CC2=CC=CC=C2CC1',
    '2-propyl tetralin': 'CCCC1CC2=CC=CC=C2CC1',
    '2-butyl tetralin': 'CCCCC1CC2=CC=CC=C2CC1',
    '2-methyl hexane': 'CCCC(C)C',
    '2-methyl heptane': 'CCCCC(C)C',
    '2-methyl octane': 'CCCCCC(C)C',
    '2-methyl nonane': 'CCCCCCC(C)C',
    '2-methyl decane': 'CCCCCCCC(C)C',
    '2-methyl undecane': 'CCCCCCCCC(C)C',
    '2-methyl dodecane': 'CCCCCCCCCC(C)C',
    '2-methyl tridecane': 'CCCCCCCCCCC(C)C',
    '2-methyl tetradecane': 'CCCCCCCCCCCC(C)C',
    '2-methyl pentadecane': 'CCCCCCCCCCCCC(C)C',
    '2-methyl hexadecane': 'CCCCCCCCCCCCCC(C)C',
    '2-methyl heptadecane': 'CCCCCCCCCCCCCCC(C)C',
    '2-methyl octadecane': 'CCCCCCCCCCCCCCCC(C)C',
    '2-methyl nonadecane': 'CCCCCCCCCCCCCCCCC(C)C',
    'n-heptane': 'CCCCCCC',
    'n-octane': 'CCCCCCCC',
    'n-nonane': 'CCCCCCCCC',
    'n-decane': 'CCCCCCCCCC',
    'n-undecane': 'CCCCCCCCCCC',
    'n-dodecane': 'CCCCCCCCCCCC',
    'n-tridecane': 'CCCCCCCCCCCCC',
    'n-tetradecane': 'CCCCCCCCCCCCCC',
    'n-pentadecane': 'CCCCCCCCCCCCCCC',
    'n-hexadecane (cetane)': 'CCCCCCCCCCCCCCCC',
    'n-heptadecane': 'CCCCCCCCCCCCCCCCC',
    'n-octadecane': 'CCCCCCCCCCCCCCCCCC',
    'methyl cyclohexane': 'CC1CCCCC1',
    'ethyl cyclohexane': 'CCC1CCCCC1',
    'propyl cyclohexane': 'CCCC1CCCCC1',
    'butyl cyclohexane': 'CCCCC1CCCCC1',
    'pentyl cyclohexane': 'CCCCCC1CCCCC1',
    'hexyl cyclohexane': 'CCCCCCC1CCCCC1',
    'heptyl cyclohexane': 'CCCCCCCC1CCCCC1',
    'octyl cyclohexane': 'CCCCCCCCC1CCCCC1',
    'nonyl cyclohexane': 'CCCCCCCCCC1CCCCC1',
    'decyl cyclohexane': 'CCCCCCCCCCC1CCCCC1',
    'undecyl cyclohexane': 'CCCCCCCCCCCC1CCCCC1',
    'Octahydropentalene': 'C1CC2CCCC2C1',
    'Hydrindane': 'C1CCC2CCCC2C1',
    'Decalin': 'C1CCC2CCCCC2C1',
    '2-methyldecalin': 'CC1CCC2CCCCC2C1',
    '2-ethyldecalin': 'CCC1CCC2CCCCC2C1',
    '2-propyldecalin': 'CCCC1CCC2CCCCC2C1',
    '2-butyldecalin': 'CCCCC1CCC2CCCCC2C1',
    '2-pentyldacalin': 'CCCCCC1CCC2CCCCC2C1'
}

for idx, row in df.iterrows():
    if row['SMILES'] == 'NOT_FOUND':
        name = row['Reference Compound'].strip()
        if name in smiles_map:
            df.at[idx, 'SMILES'] = smiles_map[name]
        else:
            print(f"Still missing: {name}")

df.to_csv(fuel_csv, index=False)
print("Updated missing SMILES")
