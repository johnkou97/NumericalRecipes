import os , sys
import numpy as np
import matplotlib.pyplot as plt
from lu_decomp import Vandermonde, ForwSub, BackSub, LUDecomp, PredictVander

def Neville(x_pred: np.ndarray, x_true: np.ndarray, y_true: np.ndarray, degree: int) -> np.ndarray:
	'''
	applies the Neville's algorithm to find the Lagrange polynomial
	for each point we use the bisection algorithm to find the m nearest neighbors 
	by finding j_low and we initialize p with these points 
	we then loop and each time we update the values of p
	at the end p[0] is the interpolated value at this point 
	'''
	m=degree+1
	nev=np.zeros_like(x_pred)
	for i,x in enumerate(x_pred):
		low=0
		up=len(x_true)-1
		while low+1<up:
			if x<x_true[int((up+low)/2)]:
				up=int((up+low)/2)
			else:
				low=int((up+low)/2)
		j_low=low-int((m-1)/2)
		if j_low<0:
			j_low=0
		if j_low+m>len(y_true):
			j_low=len(y_true)-m
		p=[y_true[i] for i in range(j_low,j_low+m)]
		for k in range(1,m):
			for j in range(m-k):
				p[j] = ((x-x_true[j_low+j+k])*p[j] + (x_true[j_low+j]-x)*p[j+1])/(x_true[j_low+j]-x_true[j_low+j+k])
		nev[i]=p[0]
	return nev


if __name__ == "__main__":
	# get the data
	data = np.genfromtxt(os.path.join(sys.path[0],"data/Vandermonde.txt"),comments='#',dtype=np.float64)
	x = data[:,0]
	y = data[:,1]
	xx = np.linspace(x[0],x[-1],1001) 	# x values to interpolate at

	# solve for the method using the vandermonde matrix like we did in the previous code
	vander = Vandermonde(x)           
	l , u = LUDecomp(vander)
	u_c = ForwSub(l,y)             
	c = BackSub(u,u_c)              
	y_van = PredictVander(xx,c)   

	y_nev = Neville(xx,x,y,len(x)-1)		# solve using the Neville's algorithm

	# plot the polynomials with the data points
	fig_1 = plt.figure()
	plt.scatter(x,y,label='data',color='black')
	plt.plot(xx,y_van,label='Vandermonde',color='orange')
	plt.plot(xx,y_nev,linestyle='--',label='Neville',color='green')
	plt.xlim(-1,101)
	plt.ylim(-400,400)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend()
	plt.savefig('./plots/compare.png', bbox_inches='tight', dpi=300)
	plt.close()

	y_2 = Neville(x,x,y,len(x)-1)       # generate predictions in the same positions with the data points

	# plot the absolute difference between predictions and actual points
	# second plot is in logscale to show the small differences in the first points
	fig_2 = plt.figure()
	plt.scatter(x,abs(y-y_2),label=r'$|y(x)-y_i|$',color='crimson')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend()
	plt.grid()
	plt.yscale('log')
	plt.savefig('./plots/nevil_dif.png', bbox_inches='tight', dpi=300)
	plt.close()