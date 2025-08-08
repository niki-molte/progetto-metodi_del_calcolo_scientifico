import json

import numpy as np
from numpy.typing import NDArray

import matplotlib
from matplotlib import cm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


import pandas as pd
pd.set_option('display.max_columns', None)


class chart():

    def __init__(self):
        pass

    def make_run_chart(self, run_stats: pd.DataFrame, trace_memory=False):

        if trace_memory:
            median_df = run_stats.groupby(['method', 'matrix', 'tol'], as_index=False)[
                ['niter', 'err', 'time', 'memu']
            ].median()
        else:
            median_df = run_stats.groupby(['method', 'matrix', 'tol'], as_index=False)[
                ['niter', 'err', 'time']
            ].median()


        methods = median_df['method'].unique()
        tolerances = sorted(median_df['tol'].unique(), reverse=True) # ordinamento crescente
        matrices = median_df['matrix'].unique()
        bar_width = 0.8 / len(tolerances)

        cmap = cm.get_cmap("viridis", len(tolerances))
        colors = [cmap(i) for i in range(len(tolerances))]

        if trace_memory:
            metrics = ['niter', 'err', 'time', 'memu']
        else:
            metrics = ['niter', 'err', 'time']

        metrics_name = ['Iterazioni', 'Errore', 'Tempo (s)', 'Memoria (MiB)']
        titles = ['Iterazioni', 'Errore', 'Tempo (s)', 'Memoria usata (MiB)']

        for matrix in matrices:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.flatten()

            if not trace_memory:
                axs[3].set_visible(False)

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
                            value = np.median(row[metric].values[:]) if row[metric].values.size > 0 else 0
                            values.append(value)
                        else:
                            values.append(0)
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

    def single_stats_solver(self):
        # carico i dati
        df_wm = self.load_stats("data/memory computation.json")
        df_nm = self.load_stats("data/computation.json")

        df = pd.concat([df_nm, df_wm[['memu']]], axis=1)

        # calcolo della mediana
        grouped = df.groupby(["method", "matrix", "tol"]).median(numeric_only=True).reset_index()

        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        resources = ["niter", "time", "err", "memu"]
        titles = {
            "niter": "Numero di Iterazioni",
            "time": "Tempo di esecuzione (s)",
            "err": "Errore",
            "memu": "Memoria (MB)"
        }

        # color map definita
        cmap = cm.get_cmap("viridis", len(tolerances))
        tol_colors = {tol: cmap(i) for i, tol in enumerate(tolerances)}

        for solver in grouped["method"].unique():
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            axs = axs.flatten()

            data_solver = grouped[grouped["method"] == solver]
            matrices = data_solver["matrix"].unique()
            x = np.arange(len(matrices))
            bar_width = 0.15

            for i, resource in enumerate(resources):
                ax = axs[i]
                shift = -1.5  # reset shift per subplot

                for j, tol in enumerate(tolerances):
                    df_plot = data_solver[data_solver["tol"] == tol]
                    heights = []

                    for matrix in matrices:
                        row = df_plot[df_plot["matrix"] == matrix]
                        if not row.empty:
                            heights.append(row[resource].values[0])
                        else:
                            heights.append(np.nan)

                    ax.bar(x + shift * bar_width, heights,
                           width=bar_width,
                           color=tol_colors[tol])
                    shift += 1

                ax.set_title(titles[resource])
                ax.set_xticks(x)
                ax.set_xticklabels(matrices)
                ax.set_yscale('log')
                ax.set_ylabel(titles[resource])

            # creazione legenda
            legend_patches = [
                mpatches.Patch(color=tol_colors[tol], label=f"tol={tol:.0e}")
                for tol in tolerances
            ]

            fig.suptitle(f"Statistiche per il solver: {solver}", fontsize=16)
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Lascia spazio sotto

            # legenda fuori grafico
            fig.legend(handles=legend_patches,
                       loc="lower center",
                       ncol=4,
                       bbox_to_anchor=(0.5, 0.02),
                       bbox_transform=fig.transFigure)

            plt.show()



    def spy(self, A: NDArray[np.float64], matrix_name: str) -> None:

        density = self.matrix_density(A)

        plt.figure(figsize=(7, 7))
        plt.spy(A)
        plt.title(f"{round(density*100, 5)}% densità di {matrix_name}")
        plt.xlabel("Colonne")
        plt.ylabel("Righe")
        plt.grid(False)
        plt.show()


    def barplot_diagonal_dominance(self, matrices: list[NDArray[np.float64]], names: list[str]) -> None:
        scores = [
            self.diagonal_dominance_percent(A) * 100
            for A in matrices
        ]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, scores, color='steelblue')

        # posizinoamento etichette
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{score:.1f}%", ha='center', va='bottom')

        plt.ylim(0, 100)
        plt.ylabel("Dominanza diagonale (%)")
        plt.title("Percentuale di righe con dominanza diagonale")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()



    @classmethod
    def diagonal_dominance_percent(self, A: NDArray[np.float64]) -> float:
        A = np.array(A)
        n = A.shape[0]
        count = 0.0

        for i in range(n):
            diag = abs(A[i, i])
            off_diag_sum = np.sum(np.abs(A[i, :])) - diag

            if diag > off_diag_sum + 1e-10:
                count += 1

        return count / n


    @classmethod
    def matrix_density(cls, A: NDArray[np.float64]) -> float:
        nonzero = np.count_nonzero(np.abs(A) > 1E-10)
        total = A.size
        return nonzero / total



