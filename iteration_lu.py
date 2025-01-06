import os , sys
import numpy as np
import matplotlib.pyplot as plt
from lu_decomp import Vandermonde, ForwSub, BackSub, LUDecomp, PredictVander
from neville import Neville

def LUIter(x_pred: np.ndarray, x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
	'''
	interpolate using the vandermonde matrix
	first we define the vandermonde matrix
	we then use the LU decomposition method to get the solution
	we iterate on the solution to get better results
	'''
	vander=Vandermonde(x)
	l , u = LUDecomp(vander)
	b = y
	u_c = ForwSub(l,b)
	c = BackSub(u,u_c)
	for i in range(n-1):
		b = np.dot(vander,c) - y
		u_c = ForwSub(l,b)
		c -= BackSub(u,u_c)
	return PredictVander(x_pred,c)


if __name__ == "__main__":
	# get the data
	data = np.genfromtxt(os.path.join(sys.path[0],"data/Vandermonde.txt"),comments='#',dtype=np.float64)
	x = data[:,0]
	y = data[:,1]
	xx = np.linspace(x[0],x[-1],1001) 	# x values to interpolate at

	y_pred_1 = LUIter(xx,x,y,1)			# generate predictions using 1 iteration

	y_pred_10 = LUIter(xx,x,y,10)		# generate predictions using 10 iterations


	# plot the polynomial with the data points
	fig_1 = plt.figure()
	plt.scatter(x,y,label='data',color='black')
	plt.plot(xx,y_pred_1,label='1 iteration',color='purple')
	plt.plot(xx,y_pred_10,linestyle='--',label='10 iterations',color='yellow')
	plt.xlim(-1,101)
	plt.ylim(-400,400)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend()
	plt.savefig('./plots/iter_comp.png', bbox_inches='tight', dpi=300)
	plt.close()

	# generate predictions in the same positions with the data points
	y_2 = LUIter(x,x,y,1)  			
	y_2_10 = LUIter(x,x,y,10)

	# plot the absolute difference between predictions and actual points
	fig_2 = plt.figure()
	plt.scatter(x,abs(y-y_2),label=r'$|y_{1}(x)-y_i|$',color='olive')
	plt.scatter(x,abs(y-y_2_10),label=r'$|y_{10}(x)-y_i|$',color='teal')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.legend()
	plt.grid()
	plt.yscale('log')
	plt.savefig('./plots/iter_dif.png', bbox_inches='tight', dpi=300)
	plt.close()