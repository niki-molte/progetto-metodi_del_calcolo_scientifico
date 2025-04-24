import json

import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import pandas as pd
pd.set_option('display.max_columns', None)

class chart():

    def __init__(self):
        pass

    def make_run_chart(self,  run_stats: pd.DataFrame):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Primo subplot
        sns.scatterplot(data=run_stats, y='niter', x='tol', hue='method', style='matrix', ax=axs[0][0])
        axs[0][0].set(xlabel="Dimensionalità", ylabel="Iterazione", title="Iterazioni vs Dimensionalità", yscale='log', xscale='log')

        # Secondo subplot
        sns.scatterplot(data=run_stats, y='err', x='tol', hue='method', style='matrix', ax=axs[0][1])
        axs[0][1].set(xlabel="Dimensionalità", ylabel="Errore", title="Errore vs Dimensionalità", yscale='log', xscale='log')

        # Terzo subplot
        sns.scatterplot(data=run_stats, y='time', x='tol', hue='method', style='matrix', ax=axs[1][0])
        axs[1][0].set(xlabel="Dimensionalità", ylabel="Tempo", title="Tempo vs Dimensionalità", yscale='log', xscale='log')

        # Nascondi il quarto subplot (vuoto)
        axs[1][1].set_visible(False)

        for ax in axs.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Prendi handles e labels da uno dei grafici
        handles, labels = axs[0, 0].get_legend_handles_labels()

        # Aggiungi una sola legenda globale nella zona in basso a sinistra
        fig.legend(handles, labels, loc='outside center', bbox_to_anchor=(0.75, 0.3), fontsize='medium', frameon=True)

        plt.tight_layout()
        plt.show()

    @classmethod
    def load_stats(cls, stats_path: str) -> pd.DataFrame:
        with open('data/computation.json') as f:  # oppure sostituisci con json.loads(...) se il JSON è in formato stringa
            data = json.load(f)

        # Prepariamo i dati in una lista di dizionari
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

        # Creiamo il DataFrame
        df = pd.DataFrame(records)
        return df


    def make_stats(self):
        df = self.load_stats("data/computation.json")

        mean_df = df.groupby(['method', 'matrix', 'tol']).mean(numeric_only=True).reset_index()
        print(mean_df)

        self.make_run_chart(mean_df)


