import numpy as np
import matplotlib.pyplot as plt
from open_int import Nx

# import the value of A that was calculated in the previous script
with open('output/open_int.txt') as f:
		lines = f.readlines()
		NORM = float(lines[0].split('=')[1])

def Distribution(x: float) -> float:
	'''
	Distribution function for the number of galaxies
	4*pi*n(x)*x^2
	'''
	return 4*np.pi*Nx(x,NORM)*(x**2)

def LCG(seed: int = 0, n: int = 10, a: int = 1103515245, c: int = 12345, m: int = 2**31) -> np.ndarray:
	'''
	random number generator 
	uses Linear Congruential Generators
	generates uniform distribution between 0 and 1
	seed: used for reproducible results
	n: number of points
	a: multiplier
	c: increment
	m: modulus
	'''
	numb = np.zeros(n)
	for i in range(n):
		seed = (a * seed + c) % m
		numb[i] = seed / m
	return numb

def Uniform(a: float = 0, b: float = 1, n: int = 1, seed: int = 42) -> np.ndarray:
	'''
	uniform number generator
	a: lower limit
	b: upper limit
	n: number of points
	seed: used for reproducible results
	'''
	return a+(b-a)*LCG(seed,n=n)

def Sampling(f: callable, a: float, b: float, N: int, seed1: int = 24, seed2: int = 42, mult: int = 100) -> np.ndarray:
	'''
	sampling using rejection sampling
	f: distribution
	a: low limit
	b: upper limit
	seed1, seed2: used for reproducible results
	'''
	n = mult*N
	
	x = Uniform(a,b,n=n,seed=seed1)
	y = Uniform(n=n,seed=seed2)
	
	max_val = np.amax(f(x))
	samp=np.zeros(N)
	m = 0
	i = 0

	while m<N:
		if i == n:
			break
		
		if y[i]<f(x[i])/max_val:
			samp[m] = x[i]
			m+=1
		i+=1

	return np.array(samp)


if __name__ == '__main__':
	
	N = 10000
	sample = np.array([Sampling(Distribution,10**-4,5,N,1312,1234),Uniform(0,2*np.pi,N),Uniform(0,np.pi,N)])
	np.save('output/sample',sample)

	x = np.linspace(10**-4,5,10000)
	y = Distribution(x)

	plt.hist(sample[0],bins=np.logspace(np.log10(10**-4),np.log10(5), 20),density=True,
		color='yellow',edgecolor='black',label='Sattelite Galaxies')
	plt.plot(x,y/100,color='red',label=r'$N(x)/ 100$')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('Relative Radius')
	plt.ylabel('Number of galaxies')
	plt.savefig('plots/distr.png', dpi=300)
	plt.close()

	plt.hist(sample[0],bins=np.logspace(np.log10(10**-4),np.log10(5), 20),density=True,
		color='yellow',edgecolor='black',label='Sattelite Galaxies')
	plt.plot(x,y/100,color='red',label=r'$N(x)/ 100$')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e-3,10)
	plt.xlim(1e-3,10)
	plt.legend()
	plt.xlabel('Relative Radius')
	plt.ylabel('Number of galaxies')
	plt.savefig('plots/distr_zoom.png', dpi=300)
	plt.close()