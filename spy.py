from core.iterative_methods import IterativeMethods
from core.chart import chart

#A = IterativeMethods.load('matrix/vem2.mtx')
mc = chart()
mc.single_stats_solver()
#mc.spy(A, 'vem2')

#spa1 = IterativeMethods.load('matrix/spa1.mtx')
#spa2 = IterativeMethods.load('matrix/spa2.mtx')
#vem1 = IterativeMethods.load('matrix/vem1.mtx')
#vem2 = IterativeMethods.load('matrix/vem2.mtx')


#matrix = [spa1, spa2, vem1, vem2]

#mc.barplot_diagonal_dominance(matrix, ['spa1', 'spa2', 'vem1', 'vem2'])

