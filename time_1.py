import os , sys
import timeit
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
	# get the data
	data = np.genfromtxt(os.path.join(sys.path[0],"data/Vandermonde.txt"),comments='#',dtype=np.float64)
	x = data[:,0]
	y = data[:,1]
	xx = np.linspace(x[0],x[-1],1001) 	# x values to interpolate at

	# time all the different ways of generating the Lagrange polynomial and print the results
	print(f'Time needed for 100 repetitions of each algorithm in seconds')

	# use 1000 repetitions then divide by 10 to find the time needed for 100 repetitions
	time_iter_1 = timeit.timeit("pred = LUIter(xx,x,y,1)", "from iteration_lu import LUIter; from __main__ import x , y, xx",number=10**3)
	time_iter_1 /= 10
	print(f'Using LU decomposition of Vandermonde matrix with 1 iteration: {time_iter_1:.4f}')

	# Neville's algorithm is much slower so we use 100 repetitions
	time_nevil = timeit.timeit("pred = Neville(xx,x,y,len(x)-1)", "from neville import Neville; from __main__ import x , y, xx",number=10**2)
	print(f'Using Neville algorithm: {time_nevil:.4f}')

	# use 1000 repetitions then divide by 10 to find the time needed for 100 repetitions
	time_iter_10 = timeit.timeit("pred = LUIter(xx,x,y,10)", "from iteration_lu import LUIter; from __main__ import x , y, xx",number=10**3)
	time_iter_10 /= 10
	print(f'Using LU decomposition of Vandermonde matrix with 10 iterations: {time_iter_10:.4f}')

	fig1 = plt.figure()
	bars=plt.barh(['LU decomposition \n1 iteration','Neville\'s algorithm','LU decomposition \n10 iterations'],[time_iter_1,time_nevil,time_iter_10],height=.5,color='magenta')
	plt.xlabel('Time [seconds]')
	plt.legend(labels = ['Time needed for\n 100 repetitions'])
	plt.bar_label(bars,fmt='%.2f')
	plt.xlim(0,18)
	plt.savefig('./plots/time.png', bbox_inches='tight', dpi=300)
	plt.close()