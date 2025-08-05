import json

import numpy as np
from numpy.typing import NDArray

import matplotlib
from matplotlib import cm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import pandas as pd
pd.set_option('display.max_columns', None)


class chart():

    def __init__(self):
        pass

    def make_run_chart(self, run_stats: pd.DataFrame):
        # Calcolo delle medie
        median_df = run_stats.groupby(['method', 'matrix', 'tol'], as_index=False)[
            ['niter', 'err', 'time', 'memu']
        ].median()

        methods = median_df['method'].unique()
        tolerances = sorted(median_df['tol'].unique(), reverse=True) # ordinamento crescente
        matrices = median_df['matrix'].unique()
        bar_width = 0.8 / len(tolerances)

        cmap = cm.get_cmap("viridis", len(tolerances))
        colors = [cmap(i) for i in range(len(tolerances))]

        metrics = ['niter', 'err', 'time', 'memu']
        metrics_name = ['Iterazioni', 'Errore', 'Tempo (s)', 'Memoria (MiB)']
        titles = ['Iterazioni', 'Errore', 'Tempo (s)', 'Memoria usata (MiB)']

        for matrix in matrices:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.flatten()
            fig.suptitle(f"Statistiche per la matrice: {matrix}", fontsize=16)

            matrix_df = median_df[median_df['matrix'] == matrix]

            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axs[i]
                x = np.arange(len(methods))

                if metric == "memu":
                    # ogni metodo rappresentato da una barra
                    values = []
                    for method in methods:
                        row = matrix_df[matrix_df['method'] == method]
                        if not row.empty:
                            # viene calcolata la media di tutti i valori di
                            # memoria usata per ciascuna tolleranza
                            value = np.mean(row[metric].values[:]) if row[metric].values.size > 0 else 0
                            values.append(value)
                        else:
                            values.append(0)
                    print(values)
                    ax.bar(x, values, width=0.6, color='gray')
                else:
                    # permette di creare tutti i grafici relativi
                    # a tutte le metriche
                    for j, tol in enumerate(tolerances):
                        tol_df = matrix_df[matrix_df['tol'] == tol]
                        values = []
                        for method in methods:
                            row = tol_df[tol_df['method'] == method]
                            if not row.empty:
                                values.append(row[metric].values[0])
                            else:
                                values.append(0)
                        ax.bar(x + j * bar_width, values, width=bar_width, color=colors[j])

                ax.set_title(title)
                ax.set_xlabel("Metodo")
                ax.set_ylabel(metrics_name[i])
                ax.set_xticks(x + bar_width * (len(tolerances) - 1) / 2 if metric != "memu" else x)
                ax.set_xticklabels(methods, rotation=45)
                ax.set_yscale('log')

            # Legenda unica
            legend_labels = [f"tol={tol:.0e}" for tol in tolerances]
            fig.legend(legend_labels, loc='lower center', ncol=len(tolerances), title="Tolleranza",
                       bbox_to_anchor=(0.5, 0.005))

            plt.tight_layout(rect=(0, 0.05, 1, 0.95))  # spazio per titolo e legenda
            plt.show()


    @classmethod
    def load_stats(cls, stats_path: str) -> pd.DataFrame:
        with open(stats_path) as f:  # oppure sostituisci con json.loads(...) se il JSON è in formato stringa
            data = json.load(f)

        # lista che conterrà tutti i dizionari
        # che esprimono i dati
        records = []

        for method, entries in data.items():
            for entry in entries:
                for matrix, metrics in entry.items():
                    record = {
                        'method': method,
                        'matrix': matrix,
                        **metrics  # aggiunge tutte le chiavi-valori di metrics
                    }
                    records.append(record)

        # creiamo il DataFrame
        df = pd.DataFrame(records)
        return df


    def make_stats(self):
        df_wm = self.load_stats("data/memory computation.json")
        df_nm = self.load_stats("data/computation.json")

        df = pd.concat([df_nm, df_wm[['memu']]], axis=1)

        self.make_run_chart(df)

    def spy(self, A: NDArray[np.float64], matrix_name: str) -> None:

        density = self.matrix_density(A)

        plt.figure(figsize=(7, 7))
        plt.title(f"{density} density of {matrix_name}")
        plt.xlabel("Colonne")
        plt.ylabel("Righe")
        plt.grid(False)
        plt.show()


    @classmethod
    def matrix_density(cls, A: NDArray[np.float64]) -> float:
        nonzero = np.count_nonzero(A)
        total = A.size
        return nonzero / total



