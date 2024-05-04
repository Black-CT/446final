import numpy as np
import pandas as pd
import pickle
import pubchempy

# # sider27
# data=pd.read_csv("data/sider27.csv")
# matrix=data.iloc[:,1:].values
# # matrix=data.iloc[:,1].values
# smiles=data["smiles"]

# # clintox
# data=pd.read_csv("data/clintox.csv")
# smile=smiles=data["smiles"]
# matrix=data.iloc[:,1:].values

# # clintox_CT_TOX.csv
# data=pd.read_csv("data/clintox_CT_TOX.csv")
# smile=smiles=data["smiles"]
# matrix=data.iloc[:,1:].values

# clintox_FDA_APPROVED.csv
data=pd.read_csv("data/clintox_FDA_APPROVED.csv")
smile=smiles=data["smiles"]
matrix=data.iloc[:,1:].values

#sider5868
# data=pd.read_csv("data/our_sider.csv")
# smile=smiles=data["SMILES"]
# matrix=data.iloc[:,4:].values

# # HIV
# data=pd.read_csv("data/HIV.csv")
# smile=smiles=data["smiles"]
# matrix=data.iloc[:,1].values

# # # BACE
# data=pd.read_csv("data/bace.csv")
# smile=smiles=data["mol"]
# matrix=data.iloc[:,2].values

# BBBP
# data=pd.read_csv("data/BBBP.csv")
# matrix=data.iloc[:,2].values
# smiles=data["smiles"]

# Lipophilicity
# data=pd.read_csv("data/Lipophilicity.csv")
# matrix=data.iloc[:,1].values
# smiles=data["smiles"]

# ESOL
# data=pd.read_csv("data/delaney-processed.csv")
# matrix=data.iloc[:,1].values
# smiles=data["smiles"]


# # Freesolv
# data=pd.read_csv("data/SAMPL.csv")
# matrix=data.iloc[:,3].values
# smiles=data["smiles"]





# def smiles_to_IUPAC(smiles):
#     compounds = pubchempy.get_compounds(smiles, namespace='smiles')
#     match = compounds[0]
#     return match.iupac_name

# # store SMILES for iupac
# dataset=[]
# IUPAC_hm = {}
# for i in range(len(smiles)):
#     try:
#         # because some drug don't have IUPAC name, these drugs' SMILES are removed.
#         if smiles_to_IUPAC(smiles[i]):
#             IUPAC_hm[smiles[i]] = smiles_to_IUPAC(smiles[i])
#             dataset.append([smiles[i], matrix[i]])
#             if i % 100 == 0: print(i)
#     except:
#         print(smiles[i])
#
# with open('drug_data/raw_iupac/clintox_CT_TOX/data.pkl', 'wb') as file:
#     pickle.dump(dataset, file)

dataset=[]
for i in range(len(smiles)):
    dataset.append([smiles[i], matrix[i]])

with open('drug_data/raw/clintox_FDA_APPROVED/data.pkl', 'wb') as file:
    pickle.dump(dataset, file)


#
# with open('drug_data/raw_iupac/sider27/SMILE_TO_IUPAC.pkl', 'wb') as file:
#     pickle.dump(IUPAC_hm, file)
# store IUPAC


# # delete drugs without IUPAC
# with open('drug_data/raw/clintox_SMILE_TO_IUPAC.pkl', 'rb') as file:
#     IUPAC_set = pickle.load(file)
#
#
# to_remove = []
# for key, val in IUPAC_set.items():
#     if val == None:
#         print(key, "-->None")
#         to_remove.append(key)
#
# for key in to_remove:
#     del IUPAC_set[key]
#
# with open('drug_data/raw/clintox_SMILE_TO_IUPAC.pkl', 'wb') as file:
#     pickle.dump(IUPAC_set, file)