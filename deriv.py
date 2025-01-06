import numpy as np
from open_int import Nx, SAT, A, B, C
from distr import NORM

def NxNorm(x: float) -> float:
	'''
	Wrapper function to include the normalization factor
	'''
	return(Nx(x,NORM))

def AnalytDeriv(x: float) -> float:
	'''
	function for the analytical derivative of n
	'''
	return NORM*SAT*((((A-3)/(B**(A-3)))*(x**(A-4))*np.exp(-((x/B)**C)))
		-(C/(B**C))*(((x/B)**(A-3))*(x**(C-1))*np.exp(-((x/B)**C))))

def CentralDif(func: callable, x: float, step: float = .1) -> float:
	'''
	derivative using central differences
	step: step for the calculation of the derivative
	'''
	return (func(x+step)-func(x-step))/(2*step)

def Ridder(func: callable, x: float, step: float = .1, d: int = 2, m: int = 5) -> float:
	'''
	derivative using ridder's algorithm
	func: function
	x: point to calculate the derivate at
	step: step for initial central difference derivative
	d: decrease factor for step
	m: maximum number of extrapolations 
	'''
	der = np.zeros(m)

	for i in range(m):
		der[i] = CentralDif(func,x,step)
		step = step/d

	prev_der = der[0]
	prev_er = 100.0
	error = 10

	for j in range(m):
		for i in range(0,m-j-1):
			der[i] = der[i+1] + (der[i+1]-der[i])/(2**(j+1) -1) 

		error = np.abs(der[0]-prev_der)

		if error-prev_er > 0:
			return prev_der	# early stopping
		
		prev_er = error
		prev_der = der[0]

	return der[0]

if __name__ == '__main__':
	
	der = Ridder(NxNorm,1,step=.01,m=10)
	print(f'numerical dn/dx = {der:.13}')

	analyt = AnalytDeriv(1)
	print(f'analytic dn/dx  = {analyt:.13}')

	print(f'abs difference  = {np.abs(analyt-der):.1}')
	
