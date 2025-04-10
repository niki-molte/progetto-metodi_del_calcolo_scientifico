import numpy as np

from core.conjugated_gradient import ConjugatedGradientMethod
from core.gauss_siedl import GaussSiedlMethod
from core.gradient import GradientMethod
from core.iterative_methods import IterativeMethods
from core.jacobi import JacobiMethod

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
