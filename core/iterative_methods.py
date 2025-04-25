import os
import json

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray

from abc import ABC, abstractmethod

from scipy.io import mmread
from scipy.linalg import issymmetric, cholesky, LinAlgError

from core.results import Results


class IterativeMethods(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """nome del metodo (da definire nella sottoclasse)"""
        pass

    @abstractmethod
    def solve(self, A: NDArray[np.float64], b: NDArray[np.float64], x_ex: NDArray[np.float64],
              n_max: int, toll: float, matrix_name: str, trace_memory: bool) -> Results:
        pass

    @staticmethod
    def load(path: str) -> NDArray:

        # controllo che il file in input esista
        # davvero, se non c'è allora viene lanciato
        # un errore
        if not os.path.isfile(path):
            raise FileNotFoundError("The specified file in path don't exists")

        matrix = mmread(path)

        if hasattr(matrix, 'todense'):
            matrix = matrix.todense()

        return np.array(matrix)

    @staticmethod
    def check_iteration(A: NDArray, x_new: NDArray, b: NDArray) -> np.float64:
        return norm(np.dot(A, x_new) - b) / norm(b)

    @staticmethod
    def evaluate_error(x_ex: NDArray, x_new: NDArray) -> np.float64:
        return norm(x_ex - x_new) / norm(x_ex)

    @staticmethod
    def converge(A: NDArray[np.float64]) -> (bool, str):
        conv = True
        msg = ''

        m, n = np.shape(A)

        # se la matrice A non è quadrata
        # allora il metodo non è applicabile
        if m != n:
            conv = False
            msg = "The input matrix should be squared"

        # se la matrice non ha le entry della
        # diagonale maggiori di 0 (1e-10)
        # scateno l'eccezione
        D = np.diag(A)
        if not np.all(D > 1e-10):
            conv = False
            msg = "The entries of the diagonal should be greater than 0.0"

        # se la matrice A non è simmetrica
        # allora il metodo non è applicabile
        # rtol è l'errore relativo
        if not issymmetric(A, rtol=1e-10):
            conv = False
            msg = "The input matrix should be symmetric"

        # se la matrice A non è definita
        # positivamente il metodo di risoluzione
        # non è applicabile. Non eseguo il controllo
        # sulla simmetria perché l'ho già fatto prima
        try:
            cholesky(A, lower=True, check_finite=True)
        except LinAlgError:
            conv = False
            msg = "The input matrix should be positivie defined"

        return conv, msg

    @staticmethod
    def save_stats(res: Results, path: str, method_name: str, matrix_name: str, trace_memory: bool):

        # se il path specificato non esiste viene 
        # lanciata l'eccezione
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find the file '{path}'")

        # carico i dati presenti nel JSON
        with open(path, 'r') as f:
            data = json.load(f)

        # creo il dizionario da salvare
        if trace_memory:
            matrix_data = {
                matrix_name: {
                    "niter": res.nit,
                    "err": res.err,
                    "tol": res.tol,
                    "time": res.tim,
                    "dim": res.dim,
                    "memu": res.mem,
                    "memp": res.mep
                }
            }
        else:
            matrix_data = {
                matrix_name: {
                    "niter": res.nit,
                    "err": res.err,
                    "tol": res.tol,
                    "time": res.tim,
                    "dim": res.dim,
                }
            }

        # aggiunge i dati nel dizionario
        # corretto
        if method_name in data:
            data[method_name].append(matrix_data)
        else:
            data[method_name] = [matrix_data]

        # salvo i dati nel JSON
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        return None
