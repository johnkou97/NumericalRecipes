import numpy as np 
from root1 import NewtonRaphson

K=1.38e-16	# erg/K
AB = 2e-13 	# cm^3 / s

def Equilibrium2(T: float, Z: float = .015, Tc: float = 10**4, psi: float = .929, 
                 nH: float = 10**0, A: float = 5*10**-10, xi: float = 10**-15) -> float:
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * ( T/1e4 )**.37 * T)*K*nH*AB + A*xi + 8.9e-26 * (T/1e4)

def F2(x: float, n: float) -> float:
	'''
	wrapper for equilibrium2
	'''
	return Equilibrium2(x, Z=.015, Tc=10**4, psi=.929, nH=n, A=5*10**-10, xi=10**-15)

def Secant(f: callable, a: float, b: float, delta: float = 1e-11, maxiter: int = 10000) -> tuple:
    '''
	root finding using secant algorithm
	f: function
	a: low bracket
	b: upper bracket
	delta: maximum relative accuracy
	maxiter: max number of iterations before stopping
	'''
    fa = f(a)
    fb = f(b)
    
    for i in range(maxiter):

        dif = ((b-a)/(fb-fa))*fb
        a = b
        b = b - dif
        fa = fb 
        fb = f(b)
        
        if abs((b-a)/b) < delta:
            return b, abs((b-a)/b), i

    raise RuntimeError("Secant algorithm did not converge.")

if __name__ == '__main__':
	
	print('Using Secant Algorithm\n')

	T_0_sec, error_0_sec, iterations_0_sec = Secant(lambda x: F2(x, n=10**0),1,10**15)

	T_pos4_sec, error_pos4_sec, iterations_pos4_sec = Secant(lambda x: F2(x, n=10**4),1,10**15)

	print(f'for nH = 10^0, T = {T_0_sec:.12}, rel error = {error_0_sec:.1}, num of iter = {iterations_0_sec}')
	print(f'for nH = 10^4, T = {T_pos4_sec:.12}, rel error = {error_pos4_sec:.1}, num of iter = {iterations_pos4_sec}')

	print('\n\nUsing Newton-Raphson Algorithm\n')

	T_neg4, error_neg4, iterations_neg4 = NewtonRaphson(lambda x: F2(x, n=10**-4), 5e14, step=1)
	T_0, error_0, iterations_0 = NewtonRaphson(lambda x: F2(x, n=10**0), 5e14, step=1)
	T_pos4, error_pos4, iterations_pos4 = NewtonRaphson(lambda x: F2(x, n=10**4), 5e14, step=1)

	print(f'for nH = 10^-4, T = {T_neg4:.12}, rel error = {error_neg4:.1}, num of iter = {iterations_neg4}')
	print(f'for nH = 10^0, T = {T_0:.12}, rel error = {error_0:.1}, num of iter = {iterations_0}')
	print(f'for nH = 10^4, T = {T_pos4:.12}, rel error = {error_pos4:.1}, num of iter = {iterations_pos4}')