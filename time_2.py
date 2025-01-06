import os , sys
import timeit
import numpy as np

if __name__ == "__main__":

	eq1_new = timeit.timeit("T_new_4, error_new_4 ,iterations_new_4 = NewtonRaphson(F1,1e4)", 
		"from root1 import NewtonRaphson,F1",number=10**5)

	eq1_fal = timeit.timeit("T_new_4, error_new_4 ,iterations_new_4 = FalsePosition(F1,1,1e7)", 
		"from root1 import FalsePosition,F1",number=10**5)	

	eq2_neg4 = timeit.timeit("T_neg4, error_neg4, iterations_neg4 = \
		NewtonRaphson(lambda x: F2(x, n=10**-4), 5e14, step=1)", 
		"from root1 import NewtonRaphson; from root2 import F2",number=10**5)

	eq2_0 = timeit.timeit("T_0_sec, error_0_sec, iterations_0_sec = Secant(lambda x: F2(x, n=10**0),1,10**15)", 
		"from root2 import Secant,F2",number=10**5)

	eq2_pos4 = timeit.timeit("T_0_sec, error_0_sec, iterations_0_sec = Secant(lambda x: F2(x, n=10**4),1,10**15)", 
		"from root2 import Secant,F2",number=10**5)

	print(f'For equation 1')
	print(f'Average time for Newton-Raphson = {eq1_new/1e5:.2e}')
	print(f'Average time for False-Position = {eq1_fal/1e5:.2e}')

	print(f'\nFor equation 2') 
	print(f'Average time for nH=1e-4 = {eq2_neg4/1e5:.2e}')
	print(f'Average time for nH=1    = {eq2_0/1e5:.2e}')
	print(f'Average time for nH=1e4  = {eq2_pos4/1e5:.2e}')