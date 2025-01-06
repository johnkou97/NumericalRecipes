import os, sys
import numpy as np
import matplotlib.pyplot as plt

def Vandermonde(x: np.ndarray) -> np.ndarray:    
    '''
    creates the vandermonde matrix
    loops over all columns and rows, calculates 
    the value and saves it to the correct position
    '''                  
    van = np.zeros((len(x),len(x)))
    for j,x_i in enumerate(x):
        for i in range(len(x)):
            van[j,i]=x_i**i  
    return van

def LUDecomp(A: np.ndarray) -> tuple:
    '''
    function to perform the LU decomposition in matrix A
    we initialize L U and then apply the Crouts algorithm
    we use matrix multiplications to avoid unnecessary loops
    '''
    N = len(A)
    L = np.zeros((N,N))
    U = np.zeros((N,N))
    for i in range(N):
        L[i,i]=1
        summ=np.dot(L[i,:i],U[:i,i:])
        U[i,i:]=A[i,i:]-summ
        summ=np.dot(L[i:,:i],U[:i,i])
        L[i:,i]=(A[i:,i]-summ)/U[i,i]
    return L,U

def ForwSub(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    '''                          
    forward substitution to be used when solving a system
    '''
    X = np.zeros_like(Y)                    
    for i in range(len(X)):
        summ = 0
        for j in range(i):
            summ += A[i,j]*X[j]             
        X[i]=(Y[i]-summ)/A[i,i]             
    return X

def BackSub(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    '''                         
    backward substitution to be used when solving a system
    '''
    X = np.zeros_like(Y)                    
    for i in range(len(X)-1,-1,-1):         # we iterate backwards
        summ = 0
        for j in range(len(X)-1,i-1,-1):    # we iterate backwards
            summ += A[i,j]*X[j]         
        X[i]=(Y[i]-summ)/A[i,i]
    return X

def PredictVander(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    '''
    generater the lagrange polynomial 
    takes as an inpute an array x to interpolate at
    and the array of coefficients c
    '''
    y_pred = np.zeros_like(x)
    for i,x_i in enumerate(x):
        for j,c_j in enumerate(c):
            y_pred[i] += c_j*(x_i**j)
    return y_pred


if __name__ == "__main__":
    # get the data
    data = np.genfromtxt(os.path.join(sys.path[0],"data/Vandermonde.txt"),comments='#',dtype=np.float64)
    x = data[:,0]
    y = data[:,1]
    xx = np.linspace(x[0],x[-1],1001)   # x values to interpolate at

    vander=Vandermonde(x)           # calculate the vandermonde matrix
    l , u = LUDecomp(vander)        # decompose it to l and u
    u_c = ForwSub(l,y)              # use forward substitution to solve l*(u_c)=y
    c = BackSub(u,u_c)              # use backward substitution to solve u*c=u_c
    print(f'c = {c}')               # print the solutions c
    y_pred = PredictVander(xx,c)    # generate the predictions

    # plot the polynomial with the data points
    fig_1 = plt.figure()
    plt.scatter(x,y,label='data',color='black')
    plt.plot(xx,y_pred,label='Lagrange polynomial',color='r')
    plt.xlim(-1,101)
    plt.ylim(-400,400)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('./plots/vander_pol.png', bbox_inches='tight', dpi=300)
    plt.close()

    y_2 = PredictVander(x,c)       # generate predictions in the same positions with the data points
    
    # plot the absolute difference between predictions and actual points
    # second plot is in logscale to show the small differences in the first points
    fig_2 = plt.figure()
    plt.scatter(x,abs(y-y_2),label=r'$|y(x)-y_i|$',color='sienna')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.savefig('./plots/vander_dif.png', bbox_inches='tight', dpi=300)
    plt.close()