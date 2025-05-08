# la libreria sys è necessaria per leggere
# gli argomenti passati all'avvio del
# programma
import os.path
import argparse

import json

import numpy as np
import pandas as pd


from core.chart import chart
from core.conjugated_gradient import ConjugatedGradientMethod
from core.gauss_siedl import GaussSiedlMethod
from core.gradient import GradientMethod
from core.iterative_methods import IterativeMethods
from core.jacobi import JacobiMethod

def parse_input():
    arg = argparse.ArgumentParser()

    # il path è un argomento NON obbligatorio
    # quando viene avviato il programma se non
    # specificato verranno risolti i sistemi
    # riguardanti spa1, spa2, vem1, vem2 presenti
    # nella cartella matrix
    arg.add_argument('--path', '-p', nargs='+', type=str, required=False,
                     help='Please specify the path of the .mtx file. The default matrix in matrix/ are: \n'
                          '    spa1 \n'
                          '    spa2 \n'
                          '    vem1 \n'
                          '    vem2')

    # il method NON è un argomento obbligatorio,
    # se non viene specificato verranno eseguiti
    # tutti i metodi per risoluzione di sistemi
    arg.add_argument('--method', '-m', nargs='+', type=str, required=False,
                     default=['jacobi', 'gauss-seidl', 'gradient', 'conjugated-gradient'],
                     help='please specify one or more iterative methods. The supported one are: \n'
                          '    jacobi \n'
                          '    gauss-seidl \n'
                          '    gradient \n'
                          '    conjugated-gradient')

    # la tolleranza NON è un argomento obbligatorio,
    # se non viene specificata allora verranno eseguiti
    # i/il metodo/i con tutte le tolleranze
    arg.add_argument('--toll', '-t', nargs='+', type=float,
                     required=False, default=[1E-4, 1E-6, 1E-8, 1E-10],
                     help='please specify one or more tolerance, type should be float. '
                          'Default values are: 1E-4, 1E-6, 1E-8, 1E-10')

    # il numero massimo di iterazioni NON è obbligatorio
    # se non specificato il valore di default è 20000
    arg.add_argument('--niter', '-ni', type=int, required=False, default=20000,
                     help='please specify number of maximum iteration, default is 20000')

    # il numero di run da eseguire per ciascun solver.
    # NON è obbligatorio, se non specificato il valore
    # di default è 1
    arg.add_argument('--nrun', '-nr', type=int, required=False, default=1,
                     help='please specify number of run for each solver, default is 1. '
                          'nrun should be positive and greather or equals 1')

    # la creazione di grafici NON è obbligatoria
    # permette di specificare se si vogliono
    # creare dei grafici che rappresentino le
    # statistiche del singolo run. di default è false
    arg.add_argument('--chart', '-c', required=False, action='store_true',
                     help='If you want some summary charts of the run set this parameter. '
                          'Default is no summary charts')

    # la creazione di grafici che contengono le
    # statistiche di tutti i run eseguiti NON
    # è obbligatoria, di default è false.
    arg.add_argument('--stats', '-s', required=False, action='store_true',
                     help='If you want some summary charts of all the run set this parameter. '
                          'Default is no stats charts.')

    # NON è obbligatorio specificare se si vuole
    # tracciare l'uso della memoria, durante l'impiego dei
    # metodi di risoluzione
    arg.add_argument('--tracememory', '-tm', required=False, action='store_true',
                     help='If you want some information about memory usage while the solver methods are running '
                          'please specify')

    # NON è obbligatorio specificare il grado di verbose
    # dell'esecuzione dei metodi, di default è false
    arg.add_argument('--verb', '-v', required=False, action='store_true',
                     help='If you want some information when the methods are running please specify '
                          'this parameter. Default is not verbose.')

    # stampa la configurazione che si sta eseguendo
    args_dict = vars(arg.parse_args())

    # non controllo se il path specificato esista
    # non controllo le iterazioni perché sono già
    # controllate nelle classi che sfruttano gli
    # argomenti
    matrix_path = args_dict.get('path')

    # se specifico un solo path viene intepretato
    # come lista di stringhe e non come lista di
    # caratteri
    if isinstance(matrix_path, str):
        matrix_path = [matrix_path]

    # genero il nome della matrice prendendo l'ultimo
    # campo dopo il backslash se ho specificato anche
    # una directory, altrimenti matrix_name è matrix_path
    matrix_name = []

    # se è stato specificato un path prendo i nomi delle
    # matrici
    if matrix_path is not None:
        for p in matrix_path:
            mn = os.path.basename(p)
            matrix_name.append(os.path.splitext(mn)[0])


    # prendo i metodi dati in input e riempio
    # SOLVERS degli oggetti necessari per
    # l'esecuzione del metodo
    solvers_from_dict = args_dict.get('method')
    solvers = []

    for solver in solvers_from_dict:

        # verifico se jacobi è nei
        # solver da usare
        if solver == "jacobi":

            # rimuovo la stringa ma aggiungo
            # l'oggetto solver
            jacobi = JacobiMethod()
            solvers.append(jacobi)

        elif solver == "gauss-seidl":

            gauss_siedl = GaussSiedlMethod()
            solvers.append(gauss_siedl)

        elif solver == "gradient":

            gradient = GradientMethod()
            solvers.append(gradient)

        elif solver == "conjugated-gradient":

            conjugated_gradient = ConjugatedGradientMethod()
            solvers.append(conjugated_gradient)

    # prendo le tolleranze date in input
    toll = args_dict.get('toll')

    # prendo il numero di iterazioni massimo
    niter = args_dict.get('niter')

    # prendo il numero di run per ciascun
    # solver
    nrun = args_dict.get('nrun')

    # controllo se andranno prodotti dei grafici
    # del run eseguito
    do_charts = args_dict.get('chart')

    # controllo se sarà necessario produrre dei
    # grafici che contengono le statistiche
    do_stats = args_dict.get('stats')

    # controllo se sarà necessario tracciare
    # l'uso della memoria
    trace_memory = args_dict.get('tracememory')

    # controllo se andranno prodotti degli
    # output durante l'esecuzione di ogni
    # fase
    verb = args_dict.get('verb')
    if verb:
        print(json.dumps(args_dict, indent=2))
    else:
        print("verbose set to False no information provided")

    return matrix_path, matrix_name, solvers, toll, niter, nrun, do_charts, do_stats, trace_memory, verb



def main(path, name, solver, tolerance, niteration, nrun, run_charts, statistics, trace_memory, verbose):

    # dataframe che contiene tutti i risultati
    # dei run eseguiti nell'istanza del programma
    if trace_memory:
        res_dataframe = pd.DataFrame(columns=['matrix', 'dim', 'method', 'niter', 'error', 'tolerance', 'time',
                                              'memory usage', 'memory peak usage'])
    else :
        res_dataframe = pd.DataFrame(
            columns=['matrix', 'dim', 'method', 'niter', 'error', 'tolerance', 'time'])

    # se è stato specificato il path calcolo la
    # matrice
    if path is not None:
        # per ogni valore di tolleranza impostato
        # viene caricata la matrice ed eseguiti i
        # solver
        for tol in tolerance:

            # per ciascuna matrice in termini di
            # nome e path
            for pt, nm in zip(path, name):

                A = IterativeMethods.load(pt)
                matrix_name = nm

                m, n = np.shape(A)
                x_ex = np.ones(shape=(m, 1))

                b = np.dot(A, x_ex)

                # per ciascun solver specificato
                # viene eseguito il metodo
                for s in solver:

                    # per ciascun numero di run definito
                    # viene iterato il metodo selezionato
                    for r in range(nrun):
                        res = s.solve(A, b, x_ex, niteration, tol, matrix_name, trace_memory)

                        if trace_memory :
                            res_dataframe.loc[len(res_dataframe)] = [matrix_name, res.dim, s.name, res.nit, res.err,
                                                                     res.tol, res.tim, res.mem, res.mep]
                        else :
                            res_dataframe.loc[len(res_dataframe)] = [matrix_name, res.dim, s.name, res.nit, res.err,
                                                                     res.tol, res.tim]

                        # se verbose è impostato vengono stampate
                        # tutte i risultati di ciascun run
                        if verbose:
                            print(f"Running {s.name} on {matrix_name} - Iteration {r + 1}/{nrun}")
                            print(res.__str__(), '\n')

    # grafici se richiesti
    if run_charts:
        make_charts = chart()
        make_charts.make_run_chart(res_dataframe)

    if statistics:
        make_charts = chart()
        make_charts.make_stats()








if __name__ == "__main__":
    mpath, mname, solvers, toll, niter, nr, do_charts, do_stats, tracemem, verb = parse_input()
    main(mpath, mname, solvers, toll, niter, nr, do_charts, do_stats, tracemem, verb)

