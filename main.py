import numpy as np

from core.conjugated_gradient import ConjugatedGradientMethod
from core.gauss_siedl import GaussSiedlMethod
from core.gradient import GradientMethod
from core.jacobi import JacobiMethod

toll = 1.e-10




solver = ConjugatedGradientMethod()

A = solver.load("matrix/vem2.mtx")
#A = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])

m, n = np.shape(A)
x_ex = np.ones(shape=(m, 1))

b = np.dot(A, x_ex)
x_0 = np.zeros(shape=(m, 1))

res = solver.solve(A, b, x_ex, 20000, toll, 'vem2')
print(res.__str__())
