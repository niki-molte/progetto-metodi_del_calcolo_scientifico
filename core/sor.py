import numpy as np
import scipy as sp

def errore(xold, xnew) -> float:
    n = np.linalg.norm(xnew - xold, ord=np.inf)
    print(f"errore -> {n:.20f}")
    return n

def triang_inf(L, b) -> np.array:
    x = np.zeros(shape=(np.shape(L)[0],1))
    x[0] = b[0]/L[0,0]
    for index in range(2,np.shape(L)[0]):
        x[index]=(b[index]-L[index, 1:index-1]*x[1:index-1])/L[index, index]


class sor:
    def solve(self, A: np.array, b: np.array, x0: np.array, nmax: int, toll: np.array, omega: float) -> float:
        L = sp.linalg.lu(A)

        B = A - L

        xold = x0
        xnew = xold + 1
        nit = 0

        n = errore(xold, xnew)

        for tol in toll:
            print(tol)
            while n > tol and nit < nmax:
                xold = xnew
                xnew = omega * (triang_inf(L, (b - B * xold))) + (1 - omega) * xold

                nit += 1

                n = errore(xnew, xold)

        return errore(xold, xnew)