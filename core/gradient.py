import time

import numpy as np
from numpy.linalg import multi_dot
from numpy._typing import NDArray

from core.iterative_methods import IterativeMethods
from core.results import Results


class GradientMethod(IterativeMethods):

    def solve(self, A: NDArray[np.float64], b: NDArray[np.float64], x_ex: NDArray[np.float64], n_max: int, toll: float,
              matrix_name: str) -> Results:

        # verifica la convergenza del metodo
        # forse meglio spostarla nel main
        conv, msg = self.converge(A)

        if not conv:
            raise ValueError(msg)

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

        # definisco il numero d'iterazioni
        # per limitarle
        nit = 0

        start = time.time()

        # applico il metodo di jacobi per calcolare la
        # soluzione del sistema.
        while self.check_iteration(A, x_new, b) > toll and nit < n_max:
            r = b - np.dot(A, x_old)
            k = np.dot(r.T, r) / np.dot(r.T, np.dot(A, r))

            # non applico il dot perché è prodotto
            # tra k (scalare) e vettore
            x_new = x_old + k * r
            x_old = x_new

            nit = nit + 1

        # salvo il tempo necessario per il
        # calcolo della convergenza
        stop = time.time()
        elapsed_time = stop - start

        # calcolo l'errore dell'ultimo run
        err = self.evaluate_error(x_ex, x_new)

        # salvo le statistiche e genero il valore
        # di ritorno della funzione
        res = Results(nit=nit, err=err, tim=elapsed_time, tol=toll)
        self.save_stats(res, "data/computation.json", "gradient", matrix_name)
        return res
