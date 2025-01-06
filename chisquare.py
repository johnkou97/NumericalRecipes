import numpy as np
import matplotlib.pyplot as plt
from open_int import Trapezoid

def ReadFile(filename: str) -> tuple:
	'''
	function to read the data from the file
	'''
	f = open(filename, 'r')
	data = f.readlines()[3:]	# Skip first 3 lines 
	nhalo = int(data[0]) 		# number of halos
	radius = []
	
	for line in data[1:]:
		if line[:-1]!='#':
			radius.append(float(line.split()[0]))
	
	radius = np.array(radius, dtype=float)    
	f.close()
	return radius, nhalo		# Return the virial radius for all the satellites in the file, and the number of halos

def F_n(x: float, norm: float, sat: float, a: float, b: float, c: float) -> float:
	'''
	function to be integrated
	we define it differently so we don't need to use an open formula
	we also keep everything as arguments so we can use it in the chi^2 function
	'''
	return 4*np.pi*norm*sat*((1/b)**(a-3))*np.exp(-(x/b)**c)*x**(a-3+2)

def Simplex3D(func: callable, n1: list = [0., 0., 0.], n2: list = [1., 0., 0.], n3: list = [0., 1., 0.], 
			n4: list = [0., 0., 1.], maxiter: int = 1000, tol: float = 1e-12) -> list:
	'''
	utilizes the downhill simplex algorithm 
	for function minimization in 3-dimensions
	input:
	func: function to be minized
	n1,n2,n3,n4: 4 initial points (N+1)
	maxiter: maximum number of iterations
	tol: sufficiently small fractional range 
	function evaluation for early termination 
	output: 
	single point that minimizes the function
	'''
	N = 3
	f_vals = np.array([func(n1),func(n2),func(n3),func(n4)],dtype=float)
	simplex = np.array([n1,n2,n3,n4],dtype=float)
	
	for iterat in range(maxiter):
		for i in range(N):
			i_min = i
			for j in range(i+1,N+1):
				if f_vals[j]<f_vals[i_min]:
					i_min = j
			if i_min != i:
				f_vals[i] , f_vals[i_min] = np.copy(f_vals[i_min]) , np.copy(f_vals[i])
				simplex[i], simplex[i_min]  = np.copy(simplex[i_min]), np.copy(simplex[i])
		if 2*abs(f_vals[-1]-f_vals[0])/abs(f_vals[-1]+f_vals[0])<tol:
			return simplex[0]
		x = np.zeros(N)
		for i in range(N):
			x[i] = np.sum(simplex[:-1,i])/N
		x_try = 2*x - simplex[-1]
		f_try = func(x_try)
		if f_try<f_vals[0]:
			x_exp = 2*x_try-x
			if func(x_exp)<f_try:
				simplex[-1] = x_exp
				f_vals[-1] = func(x_exp)
			else:
				simplex[-1] = x_try
				f_vals[-1] = f_try
		elif f_try<f_vals[-1]:
			simplex[-1] = x_try
			f_vals[-1] = f_try
		else: 
			x_try = (x+simplex[-1])/2
			f_try = func(x_try)
			if f_try<f_vals[-1]:
				simplex[-1] = x_try
				f_vals[-1] = f_try
			else:
				for i in range(1,N):
					simplex[i] = (simplex[0]+simplex[i])/2
		
	return simplex[0]

def ChiSquare(values: list, counts: list, bins: list, sat: float) -> float:
	'''
	chi^2 function
	input:
	values: list of free parameters
	counts: number of counts in each bin
	bins: the edges of the each bin
	sat: average number of galaxies per halo
	output:
	the chi^2 evaluation of our data with the model
	'''
	a = values[0]
	b = values[1]
	c = values[2]
	
	norm = sat/Trapezoid(lambda x: F_n(x,norm=1,sat=sat,a=a,b=b,c=c),0,5,100)
	
	sums = 0
	for i in range(len(bins)-1):
		mean = Trapezoid(lambda x: F_n(x,norm=norm,sat=sat,a=a,b=b,c=c),bins[i],bins[i+1],100)/sat
		sums += (counts[i]-mean)**2/(mean)
	return sums

def Plot(counts: list, bins: list, x: np.array, y: np.array, name: str = 'hist', types: str = 'png', dpi: int = 300):
	'''
	plotting function
	'''
	plt.bar(bins[:-1] + np.diff(bins) / 2, counts, np.diff(bins),color='r',label='binned data')
	plt.plot(x,y,color='black',label='fitted model')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('x')
	plt.ylabel('N')
	plt.legend()
	plt.ylim(10**-5,10**1)
	plt.savefig(f'plots/{name}.{types}',dpi=dpi)
	plt.close()

def PlotAppendix(counts: list, bins: list, x: np.array, y: np.array, name: str = 'hist-app', types: str = 'png', dpi: int = 300):
	'''
	plotting function for appendix
	'''
	plt.bar(bins[:-1] + np.diff(bins) / 2, counts, np.diff(bins),color='r',label='binned data')
	plt.plot(x,y,color='black',label='fitted model')
	plt.xlabel('x')
	plt.ylabel('N')
	plt.legend()
	plt.ylim(0,np.amax(y)+.2)
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
	count1, bins1 = np.histogram(radius1, bins=20, density=True)
	count2, bins2 = np.histogram(radius2, bins=20, density=True)
	count3, bins3 = np.histogram(radius3, bins=20, density=True)
	count4, bins4 = np.histogram(radius4, bins=20, density=True)
	count5, bins5 = np.histogram(radius5, bins=20, density=True)

	# calculating sat for each file
	sat1 = len(radius1)/nhalo1
	sat2 = len(radius2)/nhalo2
	sat3 = len(radius3)/nhalo3
	sat4 = len(radius4)/nhalo4
	sat5 = len(radius5)/nhalo5

	# minimizing chisquare using simplex
	res1=Simplex3D(lambda values: ChiSquare(values,counts=count1,bins=bins1,sat=sat1),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)

	res2=Simplex3D(lambda values: ChiSquare(values,counts=count2,bins=bins2,sat=sat2),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)
	
	res3=Simplex3D(lambda values: ChiSquare(values,counts=count3,bins=bins3,sat=sat3),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)
	
	res4=Simplex3D(lambda values: ChiSquare(values,counts=count4,bins=bins4,sat=sat4),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)
	
	res5=Simplex3D(lambda values: ChiSquare(values,counts=count5,bins=bins5,sat=sat5),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)

	np.save('output/chi_results.npy',np.array([res1,res2,res3,res4,res5]))

	# getting the model for plotting
	x = np.linspace(.0001,5,1000) 
	
	norm1 = sat1/Trapezoid(lambda x: F_n(x,sat=sat1,norm=1,a=res1[0],b=res1[1],c=res1[2]),0,5,1000)
	y1 = F_n(x,sat=sat1,norm=norm1,a=res1[0],b=res1[1],c=res1[2])/sat1

	norm2 = sat2/Trapezoid(lambda x: F_n(x,sat=sat2,norm=1,a=res2[0],b=res2[1],c=res2[2]),0,5,1000)
	y2 = F_n(x,sat=sat2,norm=norm2,a=res2[0],b=res2[1],c=res2[2])/sat2

	norm3 = sat3/Trapezoid(lambda x: F_n(x,sat=sat3,norm=1,a=res3[0],b=res3[1],c=res3[2]),0,5,1000)
	y3 = F_n(x,sat=sat3,norm=norm3,a=res3[0],b=res3[1],c=res3[2])/sat3

	norm4 = sat4/Trapezoid(lambda x: F_n(x,sat=sat4,norm=1,a=res4[0],b=res4[1],c=res4[2]),0,5,1000)
	y4 = F_n(x,sat=sat4,norm=norm4,a=res4[0],b=res4[1],c=res4[2])/sat4

	norm5 = sat5/Trapezoid(lambda x: F_n(x,sat=sat5,norm=1,a=res5[0],b=res5[1],c=res5[2]),0,5,1000)
	y5 = F_n(x,sat=sat5,norm=norm5,a=res5[0],b=res5[1],c=res5[2])/sat5

	# plotting
	Plot(count1,bins1,x,y1,'chi-fit1')
	Plot(count2,bins2,x,y2,'chi-fit2')
	Plot(count3,bins3,x,y3,'chi-fit3')
	Plot(count4,bins4,x,y4,'chi-fit4')
	Plot(count5,bins5,x,y5,'chi-fit5')

	PlotAppendix(count1,bins1,x,y1,'chi-fit-app1')
	PlotAppendix(count2,bins2,x,y2,'chi-fit-app2')
	PlotAppendix(count3,bins3,x,y3,'chi-fit-app3')
	PlotAppendix(count4,bins4,x,y4,'chi-fit-app4')
	PlotAppendix(count5,bins5,x,y5,'chi-fit-app5')

	# results
	print(f'dataset m11')
	print(f'N of sat = {sat1}')
	print(f'[a,b,c] = {res1}')
	print(f'chi^2 = {ChiSquare(values=res1,counts=count1,bins=bins1,sat=sat1)}')
	print(f'\ndataset m12')
	print(f'N of sat = {sat2}')
	print(f'[a,b,c] = {res2}')
	print(f'chi^2 = {ChiSquare(values=res2,counts=count2,bins=bins2,sat=sat2)}')
	print(f'\ndataset m13')
	print(f'N of sat = {sat3}')
	print(f'[a,b,c] = {res3}')
	print(f'chi^2 = {ChiSquare(values=res3,counts=count3,bins=bins3,sat=sat3)}')
	print(f'\ndataset m14')
	print(f'N of sat = {sat4}')
	print(f'[a,b,c] = {res4}')
	print(f'chi^2 = {ChiSquare(values=res4,counts=count4,bins=bins4,sat=sat4)}')
	print(f'\ndataset m15')
	print(f'N of sat = {sat5}')
	print(f'[a,b,c] = {res5}')
	print(f'chi^2 = {ChiSquare(values=res5,counts=count5,bins=bins5,sat=sat5)}')