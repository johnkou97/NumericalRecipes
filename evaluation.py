import numpy as np
import matplotlib.pyplot as plt
from classification import Sigmoid

def Predict(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
	'''
	generate predictions for logistic regression models
	input:
	weights: the values of the weights of the model
	X: the input array
	returns:
	predicted labels of the model
	'''
	theta = weights
	linear = np.dot(X,theta) 
	predicted_probabilities = Sigmoid(linear)
	predicted_labels = (predicted_probabilities >= 0.5).astype(int)
	return predicted_labels

def Evaluate(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
	'''
	calculates the true positives, true 
	negatives, false positives and false 
	negatives of the model
	'''
	predictions = Predict(weights, X)
	TP = TN = FP = FN = 0
	for i,pred in enumerate(predictions):
		if pred == y[i]:
			if pred == 1:
				TP +=1 
			else:
				TN +=1
		else:
			if pred == 1:
				FP +=1 
			else:
				FN +=1
	return TP, TN, FP, FN

def FScore(TP: int, FP: int, FN: int, beta: float = 1.) -> float:
	'''
	calculate the F score 
	by default beta = 1 
	'''
	prec = TP/(TP+FP)
	rec = TP/(TP+FN)
	f_score = ((1+beta**2)*prec*rec)/(beta**2 * prec + rec)
	return f_score

def Boundary(X: np.ndarray, weights: np.ndarray) -> tuple:
	'''
	creates the decision boundaries of 
	logistic regression
	'''
	x = X[:,0]
	a = weights[0]
	b = weights[1]
	return x,(-a*x)/b

def BoundaryPlot(weights: np.ndarray, X: np.ndarray, y: np.ndarray, xlim: tuple = None, 
				ylim: tuple = None, name: str = 'test', fig: str = 'png', dpi: int = 300):
	'''
	creates a plot of 2 features with the 
	decision boundary and the labels
	'''
	x_b, y_b = Boundary(X,weights)
	scatter = plt.scatter(X[:,0],X[:,1],c=y,marker='.')
	plt.legend(handles=scatter.legend_elements()[0], labels=['0','1'])
	plt.plot(x_b,y_b,color='red')
	if xlim == None:
		plt.xlim(np.amin(X[:,0]),np.amax(X[:,0]))
	else:
		plt.xlim(xlim[0],xlim[1])
	if ylim == None:
		plt.ylim(np.amin(X[:,1]),np.amax(X[:,1]))
	else:
		plt.ylim(ylim[0],ylim[1])
	plt.xlabel('Scaled Feature')
	plt.ylabel('Scaled Feature')
	plt.savefig(f'plots/{name}.{fig}',dpi=dpi)
	plt.close()


if __name__ == '__main__':
	# load data
	data = np.loadtxt('data/galaxy_data.txt').T
	y = data[-1]
	x = np.loadtxt('output/input.txt')

	# load weights
	weights_2 = np.load('output/weights-2.npy')
	weights_3 = np.load('output/weights-3.npy')
	weights_4 = np.load('output/weights-4.npy')

	# evaluations and boundary plots

	# combinations of 2 features
	Xx = x[:,0:2]
	BoundaryPlot(weights_2[0],Xx,y,name='boundary_1-2')
	tp_0, tn_0, FP_0, FN_0 = Evaluate(weights_2[0],Xx,y)
	acc_0 = (tp_0+tn_0)/len(y)
	f_0 = FScore(tp_0,FP_0,FN_0)

	Xx = np.array([x[:,0],x[:,2]]).T
	BoundaryPlot(weights_2[1],Xx,y,ylim=[-.5,1.],name='boundary_1-3')
	tp_1, tn_1, FP_1, FN_1 = Evaluate(weights_2[1],Xx,y)
	acc_1 = (tp_1+tn_1)/len(y)
	f_1 = FScore(tp_1,FP_1,FN_1)

	Xx = np.array([x[:,0],x[:,3]]).T
	BoundaryPlot(weights_2[2],Xx,y,ylim=[-.1,.1],name='boundary_1-4')
	tp_2, tn_2, FP_2, FN_2 = Evaluate(weights_2[2],Xx,y)
	acc_2 = (tp_2+tn_2)/len(y)
	f_2 = FScore(tp_2,FP_2,FN_2)

	Xx = x[:,1:3]
	BoundaryPlot(weights_2[3],Xx,y,ylim=[-.5,1.],name='boundary_2-3')
	tp_3, tn_3, FP_3, FN_3 = Evaluate(weights_2[3],Xx,y)
	acc_3 = (tp_3+tn_3)/len(y)
	f_3 = FScore(tp_3,FP_3,FN_3)

	Xx = np.array([x[:,1],x[:,3]]).T
	BoundaryPlot(weights_2[4],Xx,y,ylim=[-.1,.1],name='boundary_2-4')
	tp_4, tn_4, FP_4, FN_4 = Evaluate(weights_2[4],Xx,y)
	acc_4 = (tp_4+tn_4)/len(y)
	f_4 = FScore(tp_4,FP_4,FN_4)

	Xx = x[:,2:4]
	BoundaryPlot(weights_2[5],Xx,y,xlim=[-.5,1.],ylim=[-.1,.1],name='boundary_3-4')
	tp_5, tn_5, FP_5, FN_5 = Evaluate(weights_2[5],Xx,y)
	acc_5 = (tp_5+tn_5)/len(y)
	f_5 = FScore(tp_5,FP_5,FN_5)

	# combinations of 3 features
	Xx = x[:,0:3]
	tp_6, tn_6, FP_6, FN_6 = Evaluate(weights_3[0],Xx,y)
	acc_6 = (tp_6+tn_6)/len(y)
	f_6 = FScore(tp_6,FP_6,FN_6)

	Xx = np.array([x[:,0],x[:,1],x[:,3]]).T
	tp_7, tn_7, FP_7, FN_7 = Evaluate(weights_3[1],Xx,y)
	acc_7 = (tp_7+tn_7)/len(y)
	f_7 = FScore(tp_7,FP_7,FN_7)

	Xx = np.array([x[:,0],x[:,2],x[:,3]]).T
	tp_8, tn_8, FP_8, FN_8 = Evaluate(weights_3[2],Xx,y)
	acc_8 = (tp_8+tn_8)/len(y)
	f_8 = FScore(tp_8,FP_8,FN_8)

	Xx = x[:,1:4]
	tp_9, tn_9, FP_9, FN_9 = Evaluate(weights_3[3],Xx,y)
	acc_9 = (tp_9+tn_9)/len(y)
	f_9 = FScore(tp_9,FP_9,FN_9)

	# all 4 features
	Xx = x[:,:]
	tp_10, tn_10, FP_10, FN_10 = Evaluate(weights_4,Xx,y)
	acc_10 = (tp_10+tn_10)/len(y)
	f_10 = FScore(tp_10,FP_10,FN_10)
	
	# print results
	print(f'col  1 , 2 |TP:{tp_0:^3}|TN:{tn_0:^3}|FP:{FP_0:^3}|FN:{FN_0:^3}|accuaracy:{acc_0:^5}|F_1:{f_0:.3}')
	print(f'col  1 , 3 |TP:{tp_1:^3}|TN:{tn_1:^3}|FP:{FP_1:^3}|FN:{FN_1:^3}|accuaracy:{acc_1:^5}|F_1:{f_1:.3}')
	print(f'col  1 , 4 |TP:{tp_2:^3}|TN:{tn_2:^3}|FP:{FP_2:^3}|FN:{FN_2:^3}|accuaracy:{acc_2:^5}|F_1:{f_2:.3}')
	print(f'col  2 , 3 |TP:{tp_3:^3}|TN:{tn_3:^3}|FP:{FP_3:^3}|FN:{FN_3:^3}|accuaracy:{acc_3:^5}|F_1:{f_3:.3}')
	print(f'col  2 , 4 |TP:{tp_4:^3}|TN:{tn_4:^3}|FP:{FP_4:^3}|FN:{FN_4:^3}|accuaracy:{acc_4:^5}|F_1:{f_4:.3}')
	print(f'col  3 , 4 |TP:{tp_5:^3}|TN:{tn_5:^3}|FP:{FP_5:^3}|FN:{FN_5:^3}|accuaracy:{acc_5:^5}|F_1:{f_5:.3}')
	print(f'col  1,2,3 |TP:{tp_6:^3}|TN:{tn_6:^3}|FP:{FP_6:^3}|FN:{FN_6:^3}|accuaracy:{acc_6:^5}|F_1:{f_6:.3}')
	print(f'col  1,2,4 |TP:{tp_7:^3}|TN:{tn_7:^3}|FP:{FP_7:^3}|FN:{FN_7:^3}|accuaracy:{acc_7:^5}|F_1:{f_7:.3}')
	print(f'col  1,3,4 |TP:{tp_8:^3}|TN:{tn_8:^3}|FP:{FP_8:^3}|FN:{FN_8:^3}|accuaracy:{acc_8:^5}|F_1:{f_8:.3}')
	print(f'col  2,3,4 |TP:{tp_9:^3}|TN:{tn_9:^3}|FP:{FP_9:^3}|FN:{FN_9:^3}|accuaracy:{acc_9:^5}|F_1:{f_9:.3}')
	print(f'col 1,2,3,4|TP:{tp_10:^3}|TN:{tn_10:^3}|FP:{FP_10:^3}|FN:{FN_10:^3}|accuaracy:{acc_10:^5}|F_1:{f_10:.3}')