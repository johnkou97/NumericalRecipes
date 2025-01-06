import numpy as np
from chisquare import ReadFile, F_n, Trapezoid, Simplex3D, Plot, PlotAppendix

def LogLikelihood(values: list, counts: list, bins: list, sat: float) -> float:
	'''
	log-likelihood function
	input:
	values: list of free parameters
	counts: number of counts in each bin
	bins: the edges of the each bin
	sat: average number of galaxies per halo
	output:
	the log-likelihood evaluation of our data 
	with the model
	'''
	a = values[0]
	b = values[1]
	c = values[2]

	norm = sat/Trapezoid(lambda x: F_n(x,norm=1,sat=sat,a=a,b=b,c=c),0,5,1000)
	
	sums = 0
	for i in range(len(bins)-1):
		mean = Trapezoid(lambda x: F_n(x,norm=norm,sat=sat,a=a,b=b,c=c),bins[i],bins[i+1],100)/sat
		sums -= (counts[i]*np.log(mean)-mean)
	return sums


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
	
	res1 = Simplex3D(lambda values: LogLikelihood(values,counts=count1,bins=bins1,sat=sat1),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)

	res2 = Simplex3D(lambda values: LogLikelihood(values,counts=count2,bins=bins2,sat=sat2),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)

	res3 = Simplex3D(lambda values: LogLikelihood(values,counts=count3,bins=bins3,sat=sat3),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)
	
	res4 = Simplex3D(lambda values: LogLikelihood(values,counts=count4,bins=bins4,sat=sat4),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)
	
	res5 = Simplex3D(lambda values: LogLikelihood(values,counts=count5,bins=bins5,sat=sat5),n1=(2.4,.25,1.6)
		,n2=(2.3,.25,1.6),n3=(2.4,.35,1.6),n4=(2.4,.25,1.5),maxiter=100,tol=1e-6)

	np.save('output/pois_results.npy',np.array([res1,res2,res3,res4,res5]))


	# getting the model for plotting
	x = np.linspace(.0001,5,1000) 
	
	norm1 = sat1/Trapezoid(lambda x: F_n(x,norm=1,sat=sat1,a=res1[0],b=res1[1],c=res1[2]),0,5,1000)
	y1 = F_n(x,norm=norm1,sat=sat1,a=res1[0],b=res1[1],c=res1[2])/sat1

	norm2 = sat2/Trapezoid(lambda x: F_n(x,norm=1,sat=sat2,a=res2[0],b=res2[1],c=res2[2]),0,5,1000)
	y2 = F_n(x,norm=norm2,sat=sat2,a=res2[0],b=res2[1],c=res2[2])/sat2

	norm3 = sat3/Trapezoid(lambda x: F_n(x,norm=1,sat=sat3,a=res3[0],b=res3[1],c=res3[2]),0,5,1000)
	y3 = F_n(x,norm=norm3,sat=sat3,a=res3[0],b=res3[1],c=res3[2])/sat3

	norm4 = sat4/Trapezoid(lambda x: F_n(x,norm=1,sat=sat4,a=res4[0],b=res4[1],c=res4[2]),0,5,1000)
	y4 = F_n(x,norm=norm4,sat=sat4,a=res4[0],b=res4[1],c=res4[2])/sat4

	norm5 = sat5/Trapezoid(lambda x: F_n(x,norm=1,sat=sat5,a=res5[0],b=res5[1],c=res5[2]),0,5,1000)
	y5 = F_n(x,norm=norm5,sat=sat5,a=res5[0],b=res5[1],c=res5[2])/sat5

	# plotting
	Plot(count1,bins1,x,y1,'pois-fit1')
	Plot(count2,bins2,x,y2,'pois-fit2')
	Plot(count3,bins3,x,y3,'pois-fit3')
	Plot(count4,bins4,x,y4,'pois-fit4')
	Plot(count5,bins5,x,y5,'pois-fit5')

	PlotAppendix(count1,bins1,x,y1,'pois-fit-app1')
	PlotAppendix(count2,bins2,x,y2,'pois-fit-app2')
	PlotAppendix(count3,bins3,x,y3,'pois-fit-app3')
	PlotAppendix(count4,bins4,x,y4,'pois-fit-app4')
	PlotAppendix(count5,bins5,x,y5,'pois-fit-app5')

	# results
	print(f'dataset m11')
	print(f'[a,b,c] = {res1}')
	print(f'-ln(L) = {LogLikelihood(values=res1,counts=count1,bins=bins1,sat=sat1)}')
	print(f'\ndataset m12')
	print(f'[a,b,c] = {res2}')
	print(f'-ln(L) = {LogLikelihood(values=res2,counts=count2,bins=bins2,sat=sat2)}')
	print(f'\ndataset m13')
	print(f'[a,b,c] = {res3}')
	print(f'-ln(L) = {LogLikelihood(values=res3,counts=count3,bins=bins3,sat=sat3)}')
	print(f'\ndataset m14')
	print(f'[a,b,c] = {res4}')
	print(f'-ln(L) = {LogLikelihood(values=res4,counts=count4,bins=bins4,sat=sat4)}')
	print(f'\ndataset m15')
	print(f'[a,b,c] = {res5}')
	print(f'-ln(L) = {LogLikelihood(values=res5,counts=count5,bins=bins5,sat=sat5)}')