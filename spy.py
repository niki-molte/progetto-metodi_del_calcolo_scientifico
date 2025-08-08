from core.iterative_methods import IterativeMethods
from core.chart import chart

import numpy as np

#A = IterativeMethods.load('matrix/vem2.mtx')
mc = chart()
#mc.single_stats_solver()
#mc.spy(A, 'vem2')

spa1 = IterativeMethods.load('matrix/spa1.mtx')
spa2 = IterativeMethods.load('matrix/spa2.mtx')
vem1 = IterativeMethods.load('matrix/vem1.mtx')
vem2 = IterativeMethods.load('matrix/vem2.mtx')

print(f"condizionamento spa1 {round(np.linalg.cond(spa1, 2), 2)}")
print(f"condizionamento spa2 {round(np.linalg.cond(spa2,2), 2)}")
print(f"condizionamento vem1 {round(np.linalg.cond(vem1,2), 2)}")
print(f"condizionamento vem2 {round(np.linalg.cond(vem2,2), 2)}")


#matrix = [spa1, spa2, vem1, vem2]

#mc.barplot_diagonal_dominance(matrix, ['spa1', 'spa2', 'vem1', 'vem2'])

