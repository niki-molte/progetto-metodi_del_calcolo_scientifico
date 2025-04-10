import time

import numpy as np
from numpy._typing import NDArray

from dataclasses import field, dataclass

from core.iterative_methods import IterativeMethods
from core.results import Results


@dataclass
class JacobiMethod(IterativeMethods):

    def __init__(self):
        pass

    def solve(self, A: NDArray[np.float64], b: NDArray[np.float64], x_0: NDArray[np.float64],
              x_ex: NDArray[np.float64], n_max: int, toll: float, matrix_name: str) -> Results:

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
        x_0 = np.zeros(shape=(m, 1))
        x_new = x_0

        # definisco il numero d'iterazioni
        # per limitarle
        nit = 0

        start = time.time()

        # applico il metodo di jacobi per calcolare la
        # soluzione del sistema.
        while self.check_iteration(A, x_new, b) > toll and nit < nmax:
            x_old = x_new

            x_new = np.dot(invD, (b + np.dot(B, x_old)))
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
        self.save_stats(res, "data/computation.json", "jacobi", matrix_name)
        return res
