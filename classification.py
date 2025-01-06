import numpy as np
import math
import matplotlib.pyplot as plt

def GoldenSearch(func: callable, init1: float, init2: float, tol: float = 1e-5, maxiter: int = 1000) -> float:
	'''
	minimization routine that utilizes the golden search
	algorithm
	input: 
	func: function to find local minimum
	init1, init2: initial points 
	tol: relative precision of local minimum 
	maxiter: maximum number of iterations
	returns:
	x-value of a local minimum
	'''
	phi = (1 + math.sqrt(5))/2
	R = 1/phi
	C=1-R
	a = init1
	d = init2
	b = (a + d)/2
	if abs(d-b) > abs(b-a):
		c=b+C*(d-b)
	else:
		c=b
		b=b+C*(a-b)
	fb = func(b)
	fc= func(c)
	for i in range(maxiter):
		if abs(d-a)<tol*(abs(b)+abs(c)):
			if fb<fc:
				return b
			else:
				return c
		if fc<fb:
			a = b
			b = c
			c = R*c+C*d
			fb = fc
			fc = func(c)
		else:
			d = c
			c = b
			b = R*b+C*a
			fc = fb
			fb = func(b)
	return (b+c)/2

def BFGS(f: callable, grad_f: callable, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
	'''
	minimization routine that utilizes the Quasi-Newton
	BFGS algorithm
	input: 
	f: function to find local minimum
	grad_f: analytical derivative of f
	x0: initial point 
	tol: relative precision of local minimum 
	maxiter: maximum number of iterations
	returns:
	x-value of a local minimum
	'''
	n = len(x0)
	x = x0
	H = np.eye(n)
	cost_fun = []
	for i in range(max_iter):
		cost_fun.append(f(x))
		grad = grad_f(x)
		pk = -np.dot(H, grad)
		alpha = GoldenSearch(lambda alpha: f(x+alpha*pk), 0.0, 1.0, tol=1e-3)
		x_new = x + alpha * pk
		s = x_new - x
		if np.sqrt(np.sum((s)**2))<tol:
			return x_new, cost_fun
		y = grad_f(x_new) - grad
		rho = 1.0 / np.dot(s, y)
		A1 = np.eye(n) - rho * np.outer(s, y)
		A2 = np.eye(n) - rho * np.outer(y, s)
		H = np.dot(A1, np.dot(H, A2)) + rho * np.outer(s, s)
		x = x_new
	raise ValueError("Maximum number of iterations exceeded.")

def Sigmoid(z: np.ndarray) -> np.ndarray:
	'''
	sigmoid function
	'''
	return 1 / (1 + np.exp(-z))
	
def LossFun(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
	'''
	loss function for logistic regression
	'''
	theta = weights
	linear = np.dot(X,theta) 
	sig = Sigmoid(linear)
	eps = 1e-25 # avoid log(0)
	j = -y*np.log(sig+eps) - (1-y)*np.log(1-sig+eps)
	return np.mean(j)

def GradLoss(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    analytic derivative of the loss function for logistic regression
    '''
    dw = (1/X.shape[0])*np.dot(X.T,(Sigmoid(np.dot(X,weights))-y))
    return dw

def Fit(X: np.ndarray, y: np.ndarray, max_iter: int = 1000) -> tuple:
    '''
    logistic regression fitting
    '''
    n_features = X.shape[1]
    weights = np.ones(n_features)
    f = lambda x: LossFun(x,X,y)
    grad_f = lambda x: GradLoss(x,X,y)
    weights, cost_fun = BFGS(f,grad_f,x0=weights,max_iter=max_iter)
    return weights, cost_fun


def TrainingPlot(y: np.ndarray, name: str = 'training', fig: str = 'png', dpi: int = 300):
	'''
	function to create cost function convergance plot
	'''
	plt.figure()
	plt.plot(np.arange(len(y[0])),y[0],label='col 1-2')
	plt.plot(np.arange(len(y[1])),y[1],label='col 1-3')
	plt.plot(np.arange(len(y[2])),y[2],label='col 1-4')
	plt.plot(np.arange(len(y[3])),y[3],label='col 2-3')
	plt.plot(np.arange(len(y[4])),y[4],label='col 2-4')
	plt.plot(np.arange(len(y[5])),y[5],label='col 3-4')
	plt.plot(np.arange(len(y[6])),y[6],label='col 1-2-3')
	plt.plot(np.arange(len(y[7])),y[7],label='col 1-2-4')
	plt.plot(np.arange(len(y[8])),y[8],label='col 1-3-4')
	plt.plot(np.arange(len(y[9])),y[9],label='col 2-3-4')
	plt.plot(np.arange(len(y[10])),y[10],label='col 1-2-3-4')
	plt.ylabel('Cost Function')
	plt.xlabel('Number of iterations')
	plt.legend()
	plt.savefig(f'plots/{name}.{fig}',dpi=dpi)
	plt.close()


if __name__ == '__main__':
	#load data
	data = np.loadtxt('data/galaxy_data.txt').T
	y = data[-1]
	x = np.loadtxt('output/input.txt')

	# Training

	# combinations of 2 features
	Xx = x[:,0:2]
	weights_0, cost_fun_0=Fit(Xx,y)

	Xx = np.array([x[:,0],x[:,2]]).T
	weights_1, cost_fun_1=Fit(Xx,y)

	Xx = np.array([x[:,0],x[:,3]]).T
	weights_2, cost_fun_2=Fit(Xx,y)

	Xx = x[:,1:3]
	weights_3, cost_fun_3=Fit(Xx,y)

	Xx = np.array([x[:,1],x[:,3]]).T
	weights_4, cost_fun_4=Fit(Xx,y)

	Xx = x[:,2:4]
	weights_5, cost_fun_5=Fit(Xx,y)

	# combinations of 3 features
	Xx = x[:,0:3]
	weights_6, cost_fun_6=Fit(Xx,y)

	Xx = np.array([x[:,0],x[:,1],x[:,3]]).T
	weights_7, cost_fun_7=Fit(Xx,y)

	Xx = np.array([x[:,0],x[:,2],x[:,3]]).T
	weights_8, cost_fun_8=Fit(Xx,y)

	Xx = x[:,1:4]
	weights_9, cost_fun_9=Fit(Xx,y)

	# all 4 features
	Xx = x[:,:]
	weights_10, cost_fun_10=Fit(Xx,y)
	
	# create cost function convergance plot
	TrainingPlot([cost_fun_0,cost_fun_1,cost_fun_2,cost_fun_3,cost_fun_4
		,cost_fun_5,cost_fun_6,cost_fun_7,cost_fun_8,cost_fun_9,cost_fun_10])

	# save weights for other scripts
	np.save('output/weights-2',np.array([weights_0,weights_1,weights_2,weights_3,weights_4,weights_5]))
	np.save('output/weights-3',np.array([weights_6,weights_7,weights_8,weights_9]))
	np.save('output/weights-4',weights_10)