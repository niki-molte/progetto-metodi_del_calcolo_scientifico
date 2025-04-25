import time
import tracemalloc

import numpy as np
from numpy._typing import NDArray

from core.iterative_methods import IterativeMethods
from core.results import Results


class ConjugatedGradientMethod(IterativeMethods):

    @property
    def name(self) -> str:
        return "conjugated-gradient"

    def solve(self, A: NDArray[np.float64], b: NDArray[np.float64], x_ex: NDArray[np.float64], n_max: int, toll: float,
              matrix_name: str, trace_memory: bool) -> Results:

        # verifica la convergenza del metodo
        # forse meglio spostarla nel main
        conv, msg = self.converge(A)

        if not conv:
            raise ValueError(msg)

        # inizio a tracciare l'uso della memoria
        # da parte del metodo ma prima pulisco
        # gli stack allocati da python
        if trace_memory:
            tracemalloc.clear_traces()
            tracemalloc.start()

        # prelevo la size della matrice
        m, n = np.shape(A)

        # estrazione della diagonale e calcolo
        # della sua inversa
        D = np.diag(np.diag(A))
        invD = np.diag(1 / np.diag(A))

        # calcolo la decomposizione LU
        # sottraendo D ad A
        B = D - A

        # creo il vettore che rappresenta la
        # soluzione iniziale del sistema
        x_old = np.zeros(shape=(m, 1))
        x_new = x_old

        # calcolo il residuo iniziale
        r_old = b - np.dot(A, x_old)
        p_old = r_old

        # definisco il numero d'iterazioni
        # per limitarle
        nit = 0

        beta = 0

        start = time.time()

        # applico il metodo di jacobi per calcolare la
        # soluzione del sistema.
        while self.check_iteration(A, x_new, b) > toll and nit < n_max:
            k = np.dot(p_old.T, r_old) / np.dot(p_old.T, np.dot(A, p_old))
            x_new = x_old + (k * p_old)

            r_new = r_old - (k * (np.dot(A, p_old)))
            beta = np.dot((np.dot(A, p_old)).T, r_new) / np.dot(np.dot(A, p_old).T, p_old)

            p_new = r_new - (beta * p_old)

            x_old = x_new
            r_old = r_new
            p_old = p_new

            nit = nit + 1

        # salvo il tempo necessario per il
        # calcolo della convergenza
        stop = time.time()
        elapsed_time = stop - start

        # calcolo l'errore dell'ultimo run
        err = self.evaluate_error(x_ex, x_new)

        # ottengo l'uso di memoria da parte del
        # metodo e reinizializzo
        if trace_memory:
            usage, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # salvo le statistiche e genero il valore
            # di ritorno della funzione
            res = Results(nit=nit, err=err, tim=elapsed_time, tol=toll, dim=m, mem=usage, mep=peak)

        else:
            res = Results(nit=nit, err=err, tim=elapsed_time, tol=toll, dim=m)

        self.save_stats(res, "data/computation.json", "conjugated-gradient", matrix_name, trace_memory)
        return res
