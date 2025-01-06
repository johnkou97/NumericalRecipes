import numpy as np
import math
import  matplotlib.pyplot as plt
from open_int import Fx

NORM = 256/(5*np.pi**(3/2))

def BracketMinimum(func: callable, a: float, b: float, limit: float = 110.0, maxiter: int = 100) -> tuple:
	'''
	function that utilizes the bracketing minimum algorithm
	to return 3 numbers with f2<f1 and f2<f3, which means that
	there is at least on local minimum in [x1,x3]
	input
	func: function to find bracket minimum
	a: lower value of bracket
	b: upper value of bracket
	limit: limit for the max distance of new point
	maxiter: maximum number of iterations
	returns
	3 numbers that bracket at least one minimum
	'''
	w=(1 + math.sqrt(5))/2
	fa, fb = func(a), func(b)
	
	if fb > fa:
		a, b = b, a
		fa, fb = fb, fa
	
	c = b + (b - a) * w
	fc = func(c)

	for i in range(maxiter):
		if fc >= fb:
			return a, b, c
		
		d = b - (1/2)*((b-a)**2*(fb-fc)-(b-c)**2*(fb-fa))/((b-a)*(fb-fc)-(b-c)*(fb-fa))
		fd = func(d)
		
		if (d - b) * (d - c) < 0: #check if in between
			if fd < fc:
				a, b, c = b, d, c
				fa, fb, fc = fb, fd, fc
				return a, b, c
			elif fd>fb:
				a, b, c = a, b, d
				fa, fb, fc = fa, fb, fd
				return a, b, c
			else:
				d = c + (c-b)*w
				fd = func(d)
		elif (d-c)*(c-b)>0:  #check if it is in the right direction
			if np.abs(d - b) > limit * np.abs(c - b):
				d = c + (c - b) * w
				fd = func(d)
		else: # if not in the right direction calculate new point
			d = c + (c-b)*w
			fd = func(d)
			
		a, b, c = b, c, d
		fa, fb, fc = fb, fc, fd
	
	raise ValueError("Maximum number of iterations exceeded.")

def GoldenSearch(func: callable, init1: float, init2: float, tol: float = 1e-5, maxiter: int = 1000) -> float:
	'''
	input 
	func: function to find local minimum
	init1, init2: initial points to call bracket_minimum
	tol: relative precision of local minimum 
	maxiter: maximum number of iterations
	returns
	x-value of a local minimum
	'''
	phi = (1 + math.sqrt(5))/2
	R = 1/phi
	C=1-R
	a,b,d=BracketMinimum(func,init1,init2)
	
	if abs(d-b) > abs(b-a):
		c=b+C*(d-b)
	else:
		c=b
		b=b+C*(a-b)

	fb = func(b)
	fc= func(c)
	
	for i in range(maxiter):
		if abs(d-a)<tol*(abs(b)+abs(c)):
			if fb<fc:
				return b
			else:
				return c
		if fc<fb:
			a = b
			b = c
			c = R*c+C*d
			fb = fc
			fc = func(c)
		else:
			d = c
			c = b
			b = R*b+C*a
			fc = fb
			fb = func(b)
	raise ValueError("Maximum number of iterations exceeded.")

if __name__ == '__main__':
	# Wrapper function to include the normalization factor
    f_N = lambda x: Fx(x, NORM)

    # Find the maximum of N(x)
    maximum = GoldenSearch(lambda x: -f_N(x), 1, 2)	# Minimize -f_N(x)
    N_max = f_N(maximum)  							# Evaluate the maximum value

    # Output the results
    print(f'Local maximum at: x = {maximum:.10}')
    print(f'Maximum value of: N(x) = {N_max:.10}')

    # Plotting the results
    x = np.linspace(0.01, 5, 10000)
    y = f_N(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, color='green', label='N(x)')
    ax.scatter(maximum, N_max, color='r', marker='+', label='Maximum')
    ax.legend(loc='upper right')
    axins = ax.inset_axes([0.30, 0.30, 0.5, 0.5])
    axins.plot(x, y, color='green')
    axins.scatter(maximum, N_max, color='r', marker='+')
    axins.set_xlim(0.15, 0.30)
    axins.set_ylim(260, 270)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig('plots/maximization.png', dpi=300)
    plt.close()
