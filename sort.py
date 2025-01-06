import numpy as np
import matplotlib.pyplot as plt
from distr import Uniform

def RandomSelection(ar: np.ndarray, N: int = 100, seed: int = 12) -> np.ndarray:
	'''
	uses Fisher-Yates algorithm to randomize the array
	then selects N first elements
	ar: array to select from
	N: number of values to be selected
	seed: used for reproducible results
	'''
	array = np.copy(ar)
	leng = len(array)
	random_seed = Uniform(n=leng, seed=seed)

	for i in range(leng - 1, 0, -1):
		rand = Uniform(0,i,seed=random_seed[i])
		j = int(rand)
		array[i], array[j] = array[j], array[i]
	
	return array[:N]

def SelectionSort(a: np.ndarray, inplace: bool = True) -> np.ndarray:
	'''
	sorting by using the selection sort algorithm
	a: array to be sorted
	inplace: if False then original array remains unsorted 
	'''
	if inplace:
		ar = a
	else:
		ar = np.copy(a)

	for i in range(len(ar)-1):
		i_min = i
		for j in range(i+1,len(ar)):
			if ar[j]<ar[i_min]:
				i_min = j
		if i_min != i:
			ar[i] , ar[i_min] = ar[i_min] , ar[i]
		
	
	return ar

if __name__ == '__main__':
	
	samp = np.load('output/sample.npy')
	samp = samp[0]
	rand_samp = RandomSelection(samp, N=100, seed=2)

	rand_samp = SelectionSort(rand_samp)

	plt.plot(rand_samp,np.arange(1,101,1),color='crimson',label='Sattelite Galaxies')
	plt.xscale('log')
	plt.xlim(10**-4,5)
	plt.legend()
	plt.ylabel('Number of galaxies within radius')
	plt.xlabel('Relative Radius')
	plt.savefig('plots/rand_sort.png', dpi=300)