# la libreria sys è necessaria per leggere
# gli argomenti passati all'avvio del
# programma
import sys
import argparse

import json

import numpy as np
from Tools.scripts.verify_ensurepip_wheels import print_error

from core.conjugated_gradient import ConjugatedGradientMethod
from core.gauss_siedl import GaussSiedlMethod
from core.gradient import GradientMethod
from core.iterative_methods import IterativeMethods
from core.jacobi import JacobiMethod

def main():

    arg = argparse.ArgumentParser()

    # il path è un argomento NON obbligatorio
    # quando viene avviato il programma se non
    # specificato verranno risolti i sistemi
    # riguardanti spa1, spa2, vem1, vem2 presenti
    # nella cartella matrix
    arg.add_argument('--path', '-p', nargs='+', type=str, required=False,
                     default=['matrix/spa1.mtx',
                              'matrix/spa2.mtx',
                              'matrix/vem1.mtx',
                              'matrix/vem2.mtx'],
                     help='Please specify the path of the .mtx file. The default systems are: \n'
                          '    spa1 \n'
                          '    spa2 \n'
                          '    vem1 \n'
                          '    vem2')

    # il method NON è un argomento obbligatorio,
    # se non viene specificato verranno eseguiti
    # tutti i metodi per risoluzione di sistemi
    arg.add_argument('--method', '-m', nargs='+', type=str, required=False,
                     default=['jacobi', 'gauss-seidl', 'gradiente', 'gradiente-coniugato'],
                     help='please specify one or more iterative methods. The supported one are: \n'
                          '    jacobi \n'
                          '    gauss-seidl \n'
                          '    gradient \n'
                          '    conjugated gradient')

    # la tolleranza NON è un argomento obbligatorio,
    # se non viene specificata allora verranno eseguiti
    # i/il metodo/i con tutte le tolleranze
    arg.add_argument('--toll', '-t', nargs='+', type=float,
                     required=False, default=[1E-4, 1E-6, 1E-8, 1E-10],
                     help='please specify one or more tolerance, type should be float. '
                          'Default values are: 1E-4, 1E-6, 1E-8, 1E-10')

    # il numero massimo di iterazioni NON è obbligatorio
    # se non specificato il valore di default è 20000
    arg.add_argument('--iter', '-i', type=int, required=False, default=20000,
                     help='please specify number of maximum iteration, default is 20000')

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
    arg.add_argument('--stat', '-s', required=False, action='store_true',
                     help='If you want some summary charts of all the run set this parameter. '
                          'Default is no stats charts.')

    # NON è obbligatorio specificare il grado di verbose
    # dell'esecuzione dei metodi, di default è false
    arg.add_argument('--noverb', '-nv', required=False, action='store_true',
                     help='If you don\'t want some information when the methods are running please specify '
                          'this parameter. Default is verbose.')

    # stampa la configurazione che si sta eseguendo
    args_dict = vars(arg.parse_args())

    if args_dict.get('noverb'):
        print(json.dumps(args_dict, indent=2))
    else:
        print("verbose set to False ====> no information provided")

    sys.exit(1)

    toll = 1.e-10

    conjugated_gradient = ConjugatedGradientMethod()
    jacobi = JacobiMethod()
    gauss_siedl = GaussSiedlMethod()
    gradient = GradientMethod()

    A = IterativeMethods.load("matrix/vem2.mtx")

    m, n = np.shape(A)
    x_ex = np.ones(shape=(m, 1))

    b = np.dot(A, x_ex)
    x_0 = np.zeros(shape=(m, 1))

    solvers = [conjugated_gradient, gradient, jacobi, gauss_siedl]

    for solver in solvers:
        for _ in range(10):
            res = solver.solve(A, b, x_ex, 20000, toll, 'vem2')
            print(res.__str__())


if __name__ == "__main__":
    main()

