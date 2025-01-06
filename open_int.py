import numpy as np

SAT = 100     # number of satellite galaxies
A = 2.4       # power-law index
B = 0.25      # scale radius
C = 1.6       # exponential index

def Nx(x: float, norm: float) -> float:
    '''
    Distribution function
    '''
    return norm*SAT*((x/B)**(A-3))*np.exp(-(x/B)**C)

def Fx(x: float, norm: float) -> float:
    '''
    Function to be integrated
    '''
    return 4*np.pi*norm*SAT*((1/B)**(A-3))*np.exp(-(x/B)**C)*x**(A-3+2)

def FxNorm(x: float) -> float:
	'''
	Wrapper function to include the normalization factor
	'''
	return Fx(x, norm)

def Trapezoid(func: callable, low: float, up: float, n: int) -> float:
    '''
    Integrate using trapezoid algorithm
    func: function
    low: lower limit
    up: upper limit
    n: number of points used for calculations
    '''
    x = np.linspace(low, up, n)
    h = x[1]-x[0]
    integ = h*(func(x[0])/2+np.sum(func(x[1:-1]))+func(x[-1])/2)
    return integ

def Romberg(func: callable, low: float, up: float, n: int, m: int) -> float:
    '''
    Integrate using Romberg's algorithm
    func: function
    low: lower limit
    up: upper limit
    n: number of points used for initial calculations
    m: order of extrapolation 
    '''
    ints = np.zeros(m)
    for i in range(m):
        ints[i] = Trapezoid(func, low, up, n)
        n *= 2
    for j in range(m):
        for i in range(0, m-j-1):
            ints[i] = (4**(j + 1)*ints[i+1]-ints[i])/(4**(j+1)-1)
    return ints[0]

if __name__ == '__main__':
    # Define the range of integration
    low, up = 0, 5
    
    norm = 1  # Initial guess for normalization
    norm = 100 / Romberg(FxNorm, low, up, n=100, m=5)
    print(f'NORM = {norm:.10}')