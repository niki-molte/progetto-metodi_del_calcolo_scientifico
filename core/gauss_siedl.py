import time

import numpy as np
from numpy import tril
from numpy.typing import NDArray

from core.iterative_methods import IterativeMethods
from core.results import Results


class GaussSiedlMethod(IterativeMethods):

    def solve(self, A: NDArray[np.float64], b: NDArray[np.float64], x_ex: NDArray[np.float64], n_max: int, toll: float,
              matrix_name: str) -> Results:

        # verifica la convergenza del metodo
        # forse meglio spostarla nel main
        conv, msg = self.converge(A)

        if not conv:
            raise ValueError(msg)

        m, n = np.shape(A)

        # ricavo la matrice triangolare
        # inferiore mantenendo la
        # diagonale principale
        L = tril(A)

        # calcolo la matrice U
        # sottraendo L ad A
        B = A - L

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
        while self.check_iteration(A, x_new, b) > toll and nit < n_max:
            x_old = x_new

            x_new = self.resolve_triang_inf(L, (b - np.dot(B, x_old)))
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
        self.save_stats(res, "data/computation.json", "gauss-siedl", matrix_name)
        return res

    @classmethod
    def resolve_triang_inf(cls, L: NDArray[np.float64], b: NDArray[np.float64]):
        m, n = np.shape(L)
        x_0 = np.zeros(shape=(m, 1))

        # check che L sia davvero tril

        x_0[0] = b[0] / L[0, 0]
        for i in range(1, m):
            x_0[i] = (b[i] - np.dot(L[i, 0:i - 0], x_0[0:i - 0])) / L[i, i]

        return x_0
