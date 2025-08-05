from core.iterative_methods import IterativeMethods
from core.chart import chart

A = IterativeMethods.load('matrix/spa1.mtx')
mc = chart()

mc.spy(A, 'spa1')
