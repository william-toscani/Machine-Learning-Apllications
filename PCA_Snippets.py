import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "C:/Users/willi/OneDrive/Desktop/5ML-Riduzione-PCA-EsercizioGuidato-EsercizioAutonomia-Dati.xlsx"
df_raw = pd.read_excel(file, sheet_name=0, header=0)
df = df_raw.iloc[2:, 3:].astype(float)          # Seleziono i dati corretti del database
C = df.corr()                                   # Calcolare la matrice di correlazione
eVal, eVec = np.linalg.eig(C)                   # Calcolo autovalori e autovettori

# Ordinare gli indici degli autovalori in ordine decrescente
sorted_indices = np.argsort(eVal)[::-1]         # [::-1] inverte l'ordine (decrescente)

# Ordinare gli autovalori e gli autovettori
eVal_sorted = eVal[sorted_indices]
eVec_sorted = eVec[:, sorted_indices]

eVal_rounded = np.round(eVal_sorted, 2)
percent_weights = np.round(100 * eVal_sorted / np.sum(eVal_sorted), 2)
print(eVal_rounded)
print(percent_weights)

D=pd.DataFrame(eVec_sorted)                     # matrice degli autovettori (non ordinata)
coeff = df @ eVec[:, 1].reshape(-1, 1)
print(df @ D.to_numpy())