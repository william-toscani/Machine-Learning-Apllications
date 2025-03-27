import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate

def trova_indice(array, n):
    return next((i for i, s in enumerate(accumulate(array)) if s > (n * sum(array)/100)), len(array))

class signal:
    def __init__(self, path: str):
        self.path   = path
        self.df     = pd.read_csv(path, sep=",", header=None)
        self.df     = self.df.T                                 #tc rows: features & cols: patterns
        
    def Print(self, a=0, b=None, c=0, d=None):
        if b is None: b = self.df.shape[0]
        if d is None: d = self.df.shape[1]
        print(self.df.iloc[a:b, c:d], end="\n")
        
    def Features(self, n: int):         #Features (Rows)
        #print(self.df.iloc[n, :], end="\n")
        return self.df.iloc[n, :]

    def Pattern(self, n: int):          #Patterns (Cols)
        #print(self.df.iloc[:, n], end="\n")
        return self.df.iloc[:, n]

    def Plot(self, n: int):
        x = np.arange(0, self.df.shape[0])
        plt.plot(x, self.Pattern(n))
        
    def Custom_PCA(self, reduction=100, debug_print=False):
        
        self.C = self.df.corr()                         # Matrice di correlazione

        eVal_raw, eVec_raw = np.linalg.eig(self.C)      # Autovalori e autovettori (non ordinati)
        sorted_indices  = np.argsort(eVal_raw)[::-1]    # [::-1] inverte l'ordine (decrescente)

        self.eVal, self.eVec = eVal_raw[sorted_indices], eVec_raw[:, sorted_indices]    #Autovalori e autovettori (ordine decrescente peso %)
        self.eValPerc   = 100 * self.eVal / np.sum(self.eVal)
        
        self.D = pd.DataFrame(self.eVec)                # Matrice degli autovettori (ordinati)
        
        max_index = trova_indice(self.eValPerc, reduction)
        
        # Proiezione dei dati originali lungo tutte le componenti principali (autovettori ordinati)
        # df @ D.to_numpy() esegue il prodotto matriciale tra i dati e gli autovettori ordinati
        coeff = self.df @ self.D.to_numpy()
        self.df = coeff.iloc[:, 0:max_index]
        
        # Proiezione dei dati sulla prima componente principale (autovettore corrispondente)
        # np.reshape(-1, 1) assicura che il secondo autovettore sia una matrice colonna per la moltiplicazione
        #coeff = df @ eVec[:, 1].reshape(-1, 1)
        
        
        #print(df @ D.to_numpy())        

        if debug_print:
            print(f"Matrice di correlazione: {np.round(self.C, 2)}", end="\n\n")
            #print(f"Lista autovalori disordinati {np.round(eVal_raw,2)}", end="\n\n")
            #print(f"Lista autovettori disordinati {np.round(eVec_raw,2)}", end="\n\n")
            print(f"Lista autovalori ordinati {np.round(self.eVal,2)}", end="\n\n")
            print(f"Lista autovettori ordinati {np.round(self.eVec,2)}", end="\n\n")
            print(f"Matrice degli autovettori ordinati: {np.round(self.D, 2)}", end="\n\n")

b0 = signal("C:/Users/willi/OneDrive/Desktop/signal_b_g0.csv")
b1 = signal("C:/Users/willi/OneDrive/Desktop/signal_b_g1.csv")

b0.Custom_PCA(75)
b1.Custom_PCA(75)


