import numpy as np
import matplotlib.pyplot as plt
from generate import Generate

def CooleyTukeyFFT(inp: np.ndarray) -> np.ndarray:
	'''
	calculates the fft of an input signal
	using the Cooley-Tukey algorithm
	Uses recursion to calculate the fft
	'''
	x = np.copy(inp)
	n = len(x)
	if n == 1:
		return x
	even = CooleyTukeyFFT(x[::2])
	odd = CooleyTukeyFFT(x[1::2])
	fft = np.zeros(n, dtype=np.complex128)
	for k in range(n // 2):
		fft[k] = even[k] + np.exp(-2j * np.pi * k / n) * odd[k]
		fft[k + n // 2] = even[k] - np.exp(-2j * np.pi * k / n) * odd[k]
	return fft

def CooleyTukeyIFFT(inp: np.ndarray) -> np.ndarray:
	'''
	calculates the inverse fft of an input
	signal using the Cooley-Tukey algorithm
	Uses recursion to calculate the ifft
	'''
	x = np.copy(inp)
	n = len(x)
	if n == 1:
		return x
	even = CooleyTukeyIFFT(x[::2])
	odd = CooleyTukeyIFFT(x[1::2])
	ifft = np.zeros(n, dtype=np.complex128)
	for k in range(n // 2):
		ifft[k] = even[k] + np.exp(+2j * np.pi * k / n) * odd[k]
		ifft[k + n // 2] = even[k] - np.exp(+2j * np.pi * k / n) * odd[k]
	return ifft / 2

def FFT3D(values: np.ndarray) -> np.ndarray:
	'''
	calculates the 3D fft of an input using 
	the Cooley-Tukey algorithm
	Use the CooleyTukeyFFT function to calculate the fft
	'''
	f1 = np.zeros_like(values,dtype=np.complex128)
	f2 = np.zeros_like(values,dtype=np.complex128)
	for i in range(len(values)):
		flat = np.array(values[:,:,i],dtype=np.complex128)
		for j in range(len(values)):
			flat[j,:] = CooleyTukeyFFT(flat[j,:])
		for j in range(len(values)):
			flat[:,j] = CooleyTukeyFFT(flat[:,j])
		f1[:,:,i] = flat        
	for i in range(len(values)):
		for j in range(len(values)):
			f2[i,j,:] = CooleyTukeyFFT(f1[i,j,:])
	return f2
	
def IFFT3D(values: np.ndarray) -> np.ndarray:
	'''
	calculates the 3D inverse fft of an input 
	using the Cooley-Tukey algorithm
	Use the CooleyTukeyIFFT function to calculate the ifft
	'''
	f1 = np.zeros_like(values,dtype=np.complex128)
	f2 = np.zeros_like(values,dtype=np.complex128)
	for i in range(len(values)):
		flat = np.array(values[:,:,i],dtype=np.complex128)
		for j in range(len(values)):
			flat[j,:] = CooleyTukeyIFFT(flat[j,:])
		for j in range(len(values)):
			flat[:,j] = CooleyTukeyIFFT(flat[:,j])
		f1[:,:,i] = flat        
	for i in range(len(values)):
		for j in range(len(values)):
			f2[i,j,:] = CooleyTukeyIFFT(f1[i,j,:])
	return f2

if __name__ == '__main__':
	# generate the densities and calculate the contrast
	densities, grid = Generate()
	rho_aver = 1024/16**3
	contrast = (densities-rho_aver)/(rho_aver)

	# fft of contrast densities
	fourier = FFT3D(contrast)

	# getting k
	N = 16
	kx = np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0))) 
	ky = np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0))) 
	kz = np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0))) 
	KX, KY, KZ = np.meshgrid(kx, ky,kz)
	
	# calculate the fourier potential
	small_value = 1e-10 # avoid division by 0
	fourier = -fourier/(KX**2 + KY**2+KZ**2+small_value) 

	# calculate the potential with inverse fft
	potential = np.real(IFFT3D(fourier))

	# create the plots
	plot_ind = [4,9,11,14]
	X, Y = np.meshgrid(grid,grid)
	for i in plot_ind:
		fig = plt.figure()
		plt.pcolormesh(X,Y,np.log10(np.abs(fourier[:,:,i])),cmap='inferno')
		plt.colorbar()
		plt.savefig(f'plots/fourier_{i+.5}.png', dpi=300)
		plt.close() 

		fig = plt.figure()
		plt.pcolormesh(X,Y,potential[:,:,i],cmap='inferno_r')
		plt.colorbar()
		plt.savefig(f'plots/potential_{i+.5}.png', dpi=300)
		plt.close() 

		