import numpy as np
import matplotlib.pyplot as plt
from chisquare import ReadFile, F_n
from open_int import Romberg
from scipy.special import gammainc

def PopulateBins(counts: np.ndarray, bins: np.ndarray, sat: int, res: np.ndarray, m: int = 4) -> np.ndarray:
	'''
	populate the bins with the model
	'''
	norm = sat/Romberg(lambda x: F_n(x,norm=1,sat=sat,a=res[0],b=res[1],c=res[2]),0,5,100,m=m)
	y = np.zeros_like(counts)
	for i in range(len(bins)-1):
		y[i]=(Romberg(lambda x: F_n(x,norm=norm,sat=sat,a=res[0],b=res[1],c=res[2])
			,bins[i],bins[i+1],100,m=3))
	return y

def GTest(counts: np.ndarray, y: np.ndarray) -> tuple:
	'''
	input:
	counts: number of counts in each bin
	y: fitted model evaluated at the center 
	of each bin
	output:
	G: the G value
	P: the probability of the G-test
	'''
	sums = 0
	for i in range(len(y)):
		if y[i] !=0 :  # avoiding division by zero
			sums += counts[i]*np.log(counts[i]/y[i])
	G = 2*sums
	k = len(counts)-3
	P = gammainc(k/2,G/2)
	
	return G,P

def KSTest(counts: np.ndarray, y: np.ndarray) -> tuple:
	'''
	input:
	counts: number of counts in each bin
	y: fitted model evaluated at the center 
	of each bin
	output:
	cumul_counts: the cdf of the data
	cumul_y: the cdf of the model
	P_ks: the Probability of K-S test
	pos: position of max difference
	'''
	if len(counts)!=len(y):
		raise ValueError("Length of counts must be equal to length of y")

	cumul_counts=np.zeros_like(counts)
	cumul_counts[0]=counts[0]
	for i in range(1,len(counts)):
		cumul_counts[i]=cumul_counts[i-1]+counts[i]
	cumul_counts = cumul_counts/cumul_counts[-1]  #normalize
	
	cumul_y=np.zeros_like(y)
	cumul_y[0]=y[0]
	for i in range(1,len(y)):
		cumul_y[i]=cumul_y[i-1]+y[i]
	cumul_y = cumul_y/cumul_y[-1]
	
	ks=np.amax(np.abs(cumul_counts-cumul_y))
	pos=np.argmax(np.abs(cumul_counts-cumul_y)) #normalize
	
	N = len(counts)
	ks = (np.sqrt(N)+.12+.11/np.sqrt(N))*ks

	if ks < 1.18:
		exp = np.exp(-np.pi**2/(8*ks**2))
		P_ks = ((np.sqrt(2*np.pi))/ks)*(exp+exp**9+exp**25)
	else:
		exp = np.exp(-2*ks**2)
		P_ks = 1 - 2*(exp-exp**4+exp**9)
	return cumul_counts, cumul_y,P_ks,pos

def KSPlot(bins: np.ndarray, cumul_counts: np.ndarray, cumul_y: np.ndarray, x: np.ndarray, 
		pos: int, name: str = 'test', types: str = 'png', dpi: int = 300):
	'''
	plotting function
	'''
	plt.bar(bins[:-1] + np.diff(bins) / 2, cumul_counts, np.diff(bins),color='blue',label='data cdf')
	plt.plot(x,cumul_y,color='green',label='model cdf')
	plt.scatter(x,cumul_y,color='green')
	plt.vlines(x[pos],cumul_counts[pos],cumul_y[pos],color='red',label='max difference')
	plt.legend()
	plt.savefig(f'plots/{name}.{types}',dpi=dpi)
	plt.close()

if __name__ == '__main__':
	
	#reading the data
	radius1, nhalo1 = ReadFile('data/satgals_m11.txt')
	radius2, nhalo2 = ReadFile('data/satgals_m12.txt')
	radius3, nhalo3 = ReadFile('data/satgals_m13.txt')
	radius4, nhalo4 = ReadFile('data/satgals_m14.txt')
	radius5, nhalo5 = ReadFile('data/satgals_m15.txt')

	# binning the data
	count1, bins1 = np.histogram(radius1, bins=20)
	count2, bins2 = np.histogram(radius2, bins=20)
	count3, bins3 = np.histogram(radius3, bins=20)
	count4, bins4 = np.histogram(radius4, bins=20)
	count5, bins5 = np.histogram(radius5, bins=20)

	# calculating sat for each file
	# this time sat = len(radius)
	sat1 = len(radius1)
	sat2 = len(radius2)
	sat3 = len(radius3)
	sat4 = len(radius4)
	sat5 = len(radius5)

	# reading the results of parameters after fitting
	chi_res = np.load('output/chi_results.npy')
	pois_res = np.load('output/pois_results.npy')

	# getting the models
	x1 = bins1[:-1] + np.diff(bins1) / 2
	x2 = bins2[:-1] + np.diff(bins2) / 2
	x3 = bins3[:-1] + np.diff(bins3) / 2
	x4 = bins4[:-1] + np.diff(bins4) / 2 
	x5 = bins5[:-1] + np.diff(bins5) / 2

	y1_chi = PopulateBins(count1,bins1,sat1,chi_res[0],m=4)
	y2_chi = PopulateBins(count2,bins2,sat2,chi_res[1],m=3)
	y3_chi = PopulateBins(count3,bins3,sat3,chi_res[2],m=3)
	y4_chi = PopulateBins(count4,bins4,sat4,chi_res[3],m=4)
	y5_chi = PopulateBins(count5,bins5,sat5,chi_res[4],m=4)

	y1_pois = PopulateBins(count1,bins1,sat1,pois_res[0],m=5)
	y2_pois = PopulateBins(count2,bins2,sat2,pois_res[1],m=4)
	y3_pois = PopulateBins(count3,bins3,sat3,pois_res[2],m=4)
	y4_pois = PopulateBins(count4,bins4,sat4,pois_res[3],m=4)
	y5_pois = PopulateBins(count5,bins5,sat5,pois_res[4],m=4)

	# G-test
	G1_chi, PG1_chi = GTest(count1,y1_chi)
	G2_chi, PG2_chi = GTest(count2,y2_chi)
	G3_chi, PG3_chi = GTest(count3,y3_chi)
	G4_chi, PG4_chi = GTest(count4,y4_chi)
	G5_chi, PG5_chi = GTest(count5,y5_chi)

	G1_pois, PG1_pois = GTest(count1,y1_pois)
	G2_pois, PG2_pois = GTest(count2,y2_pois)
	G3_pois, PG3_pois = GTest(count3,y3_pois)
	G4_pois, PG4_pois = GTest(count4,y4_pois)
	G5_pois, PG5_pois = GTest(count5,y5_pois)

	# k-s test
	cumul_counts1_chi, cumul_y1_chi,ks1_chi,pos1_chi = KSTest(count1,y1_chi)
	cumul_counts2_chi, cumul_y2_chi,ks2_chi,pos2_chi = KSTest(count2,y2_chi)
	cumul_counts3_chi, cumul_y3_chi,ks3_chi,pos3_chi = KSTest(count3,y3_chi)
	cumul_counts4_chi, cumul_y4_chi,ks4_chi,pos4_chi = KSTest(count4,y4_chi)
	cumul_counts5_chi, cumul_y5_chi,ks5_chi,pos5_chi = KSTest(count5,y5_chi)

	cumul_counts1_pois, cumul_y1_pois,ks1_pois,pos1_pois = KSTest(count1,y1_pois)
	cumul_counts2_pois, cumul_y2_pois,ks2_pois,pos2_pois = KSTest(count2,y2_pois)
	cumul_counts3_pois, cumul_y3_pois,ks3_pois,pos3_pois = KSTest(count3,y3_pois)
	cumul_counts4_pois, cumul_y4_pois,ks4_pois,pos4_pois = KSTest(count4,y4_pois)
	cumul_counts5_pois, cumul_y5_pois,ks5_pois,pos5_pois = KSTest(count5,y5_pois)

	# plots for k-s test
	KSPlot(bins1,cumul_counts1_chi,cumul_y1_chi,x1,pos1_chi,name='ks_chi_1')
	KSPlot(bins2,cumul_counts2_chi,cumul_y2_chi,x2,pos2_chi,name='ks_chi_2')
	KSPlot(bins3,cumul_counts3_chi,cumul_y3_chi,x3,pos3_chi,name='ks_chi_3')
	KSPlot(bins4,cumul_counts4_chi,cumul_y4_chi,x4,pos4_chi,name='ks_chi_4')
	KSPlot(bins5,cumul_counts5_chi,cumul_y5_chi,x5,pos5_chi,name='ks_chi_5')

	KSPlot(bins1,cumul_counts1_pois,cumul_y1_pois,x1,pos1_pois,name='ks_pois_1')
	KSPlot(bins2,cumul_counts2_pois,cumul_y2_pois,x2,pos2_pois,name='ks_pois_2')
	KSPlot(bins3,cumul_counts3_pois,cumul_y3_pois,x3,pos3_pois,name='ks_pois_3')
	KSPlot(bins4,cumul_counts4_pois,cumul_y4_pois,x4,pos4_pois,name='ks_pois_4')
	KSPlot(bins5,cumul_counts5_pois,cumul_y5_pois,x5,pos5_pois,name='ks_pois_5')

	# printing the results
	print(f'dataset m11')
	print(f'chi^2 G = {G1_chi}')
	print(f'chi^2 G test Q = {1-PG1_chi}')
	print(f'poisson G = {G1_pois}')
	print(f'poisson G test Q = {1-PG1_pois}')
	print(f'chi^2 k-s test Q = {1-ks1_chi}')
	print(f'poisson k-s Q = {1-ks1_pois}')
	print(f'\ndataset m12')
	print(f'chi^2 G = {G2_chi}')
	print(f'chi^2 G test Q = {1-PG2_chi}')
	print(f'poisson G = {G2_pois}')
	print(f'poisson G test Q = {1-PG2_pois}')
	print(f'chi^2 k-s test Q = {1-ks2_chi}')
	print(f'poisson k-s test Q = {1-ks2_pois}')
	print(f'\ndataset m13')
	print(f'chi^2 G = {G3_chi}')
	print(f'chi^2 G test Q = {1-PG3_chi}')
	print(f'poisson G = {G3_pois}')
	print(f'poisson G test Q = {1-PG3_pois}')
	print(f'chi^2 k-s test Q = {1-ks3_chi}')
	print(f'poisson k-s test Q = {1-ks3_pois}')
	print(f'\ndataset m14')
	print(f'chi^2 G = {G4_chi}')
	print(f'chi^2 G test Q = {1-PG4_chi}')
	print(f'poisson G = {G4_pois}')
	print(f'poisson G test Q = {1-PG4_pois}')
	print(f'chi^2 k-s test Q = {1-ks4_chi}')
	print(f'poisson k-s test Q = {1-ks4_pois}')
	print(f'\ndataset m15')
	print(f'chi^2 G = {G5_chi}')
	print(f'chi^2 G test Q = {1-PG5_chi}')
	print(f'poisson G = {G5_pois}')
	print(f'poisson G test Q = {1-PG5_pois}')
	print(f'chi^2 k-s test Q = {1-ks5_chi}')
	print(f'poisson k-s test Q = {1-ks5_pois}')