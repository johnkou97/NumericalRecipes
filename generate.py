import numpy as np
import matplotlib.pyplot as plt

def Generate() -> tuple:
	'''
	generates the interpolated densities of 1024 particles
	in a 3D 16x16x16 grid each with a mass of 1 unit
	returns:
	densities: 3D array with the interpolated densities
	grid: 1D array with the coordinates 
	'''
	np.random.seed(121)
	n_mesh = 16
	n_part = 1024
	positions = np.random.uniform(low=0, high=n_mesh, size=(3, n_part))
	grid = np.arange(n_mesh) + 0.5
	densities = np.zeros(shape=(n_mesh, n_mesh, n_mesh))
	cellvol = 1.
	for p in range(n_part):
		cellind = np.zeros(shape=(3, 2))
		dist = np.zeros(shape=(3, 2))
		for i in range(3):
			cellind[i] = np.where((abs(positions[i, p] - grid) < 1) |
								  (abs(positions[i, p] - grid - 16) < 1) | 
								  (abs(positions[i, p] - grid + 16) < 1))[0]
			dist[i] = abs(positions[i, p] - grid[cellind[i].astype(int)])
		cellind = cellind.astype(int)
		for (x, dx) in zip(cellind[0], dist[0]):    
			for (y, dy) in zip(cellind[1], dist[1]):
				for (z, dz) in zip(cellind[2], dist[2]):
					if dx > 15: dx = abs(dx - 16)
					if dy > 15: dy = abs(dy - 16)
					if dz > 15: dz = abs(dz - 16)
					densities[x, y, z] += (1 - dx)*(1 - dy)*(1 - dz) / cellvol
	return densities, grid  

if __name__ == '__main__':
	# generate the densities and calculate the contrast
	densities, grid = Generate()
	rho_aver = 1024/16**3
	contrast = (densities-rho_aver)/rho_aver

	# create the plots
	plot_ind = [4,9,11,14]
	X, Y = np.meshgrid(grid,grid)
	for i in plot_ind:
		fig = plt.figure()
		plt.pcolormesh(X,Y,contrast[:,:,i],cmap='inferno')
		plt.colorbar()
		plt.savefig(f'plots/2d_slice_{i+.5}.png', dpi=300)
		plt.close()
	