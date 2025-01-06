if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt
	
	# get data from file
	data = np.loadtxt('data/galaxy_data.txt').T
	x = data[:-1] 		# inputs
	y = data[-1] 		# labels

	# feature scaling (mean=0, std=1)
	# plus create plots
	for i in range(len(data)-1):
		x[i] = (x[i] - x[i].mean()) / x[i].std()

		fig = plt.figure()	
		plt.hist(x[i],bins=20,color='crimson',edgecolor='black')
		plt.yscale('log')
		plt.xlabel('Scaled Feature')
		plt.ylabel('N')
		plt.savefig(f'plots/distr_{i}.png')
		plt.close()

	# reverse it for shape [m x n]
	x = x.T

	# save in .txt file
	np.savetxt('output/input.txt',x)