import numpy as np
from deriv import CentralDif

# here no need for nH nor ne as they cancel out, we also cancel out k
def Equilibrium1(T: float, Z: float, Tc: float, psi: float) -> float:
	'''
	Equilibrium equation 1
	'''
	return psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T

def F1(x: float) -> float:
	'''
	wrapper for equilibrium1
	'''
	return Equilibrium1(x, Z=.015, Tc=10**4, psi=.929)

def FalsePosition(f: callable, a: float, b: float, delta: float = 1e-11, maxiter: int = 10000) -> tuple:
	'''
	root finding using false-position algorithm
	f: function
	a: low bracket
	b: upper bracket
	delta: maximum relative accuracy
	maxiter: max number of iterations before stopping
	'''

	fa = f(a)
	fb = f(b)

	if fa * fb > 0:
		raise ValueError("Same sign at both ends of the interval.")

	for i in range(maxiter):
		c = (a * fb - b * fa) / (fb - fa)
		fc = f(c)

		if fa * fc < 0:
			b = c
			fb = fc
		else:
			a = c
			fa = fc

		if (b-a)/c < delta:
			return c, (b-a)/c, i

	raise RuntimeError("False Position method did not converge.")

def NewtonRaphson(f: callable, a: float, delta: float = 1e-11, step: float = .1, maxiter: int = 10000) -> tuple:
	'''
	root finding using Newton-Raphson algorithm
	f: function
	a: starting point
	delta: maximum relative accuracy
	step: step for the central difference algorithm
	maxiter: max number of iterations before stopping
	'''
	for i in range(maxiter):
		fa = f(a)
		der = CentralDif(f,a,step=step)
		dx = fa / der
		a -= dx
		if abs(dx/a) < delta:
			return a, abs(dx/a), i
	raise RuntimeError("Newton-Raphson algorithm did not converge.")


if __name__ == '__main__':
	
	low, up = 1, 10**7
	print('Using False-Position\n')
	T_fal, error_fal ,iterations_fal = FalsePosition(F1,low,up)

	print(f'T = {T_fal:.12}')
	print(f'Accuracy = {error_fal:.1}')
	print(f'number of iterations = {iterations_fal}')


	print('\n\nUsing Newton-Raphson\n')

	a = 2e5
	print(f'Starting from a = {a:.1e}')
	T_new_5, error_new_5 ,iterations_new_5 = NewtonRaphson(F1,a)

	print(f'T = {T_new_5:.12}')
	print(f'Accuracy = {error_new_5:.1}')
	print(f'number of iterations = {iterations_new_5}')

	a = 1e4
	print(f'\nStarting from a = {a:.1e}')
	T_new_4, error_new_4 ,iterations_new_4 = NewtonRaphson(F1,a)

	print(f'T = {T_new_4:.12}')
	print(f'Accuracy = {error_new_4:.1}')
	print(f'number of iterations = {iterations_new_4}')