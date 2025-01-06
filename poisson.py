import numpy as np

def Poisson32(l: float, k: int) -> float:
    '''
    we calculate the poisson probability given l and k 
    we make sure we operate with 32bit numbers in every step
    when we multiply 32bit int with float the output is a 64bit float
    so we convert int32 to float32 before making the operations
    result is float32
    '''
    l = np.float32(l)                         
    k = np.int32(k)
    ans = np.float32(1)                      
    for i in range(k,0,-1):
        x = np.float32(i)                   
        ans *= l/x                            
    return ans*np.exp(-l)                   


if __name__ == "__main__":
    # we define the pairs of lamda,k and print the output of our poisson function
    l , k = 1 , 0
    print(f'P({l,k}) = {Poisson32(l,k):.6E}')

    l ,k = 5 ,  10
    print(f'P({l,k}) = {Poisson32(l,k):.6E}')

    l , k = 3 , 21
    print(f'P({l,k}) = {Poisson32(l,k):.6E}')

    l , k = 2.6 , 40
    print(f'P({l,k}) = {Poisson32(l,k):.6E}')

    l , k = 101 , 200
    print(f'P({l,k}) = {Poisson32(l,k):.6E}')
