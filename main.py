import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import math

def setup_b(order, n_der):
    b = np.zeros((order))
    b[n_der] = 1.0
    return b

def setup_A(x, x0):
    A = np.zeros((x.shape[0], x.shape[0]))

    for pt_idx, point in enumerate(x):
        dx = point - x0
        
        for o in range(x.shape[0]):
            A[o, pt_idx] = dx**o / sp.factorial(o)
    
    return A
    
def get_deriv(x, y, x0, n_der):
    A = setup_A(x, x0)
    b = setup_b(x.shape[0],n_der)
    w = np.linalg.solve(A,b)

    return np.dot(y,w)


import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt

def setup_b(order, n_der):
    b = np.zeros((order))
    b[n_der] = 1.0
    return b

def setup_A(x, x0):
    A = np.zeros((x.shape[0], x.shape[0]))

    for pt_idx, point in enumerate(x):
        dx = point - x0
        
        for o in range(x.shape[0]):
            A[o, pt_idx] = dx**o / sp.factorial(o)
    
    return A
    
def get_deriv(x, y, x0, n_der):
    A = setup_A(x, x0)
    b = setup_b(x.shape[0],n_der)
    w = np.linalg.solve(A,b)

    return np.dot(y,w)

def calc_d(x, y):
    d     = np.zeros(x.shape)
    d[0]  = get_deriv(x[:3], y[:3],x[0],1)
    d[-1] = get_deriv(x[-3:], y[-3:],x[-1],1)
    
    for i in range(1,x.shape[0]-1):
        d[i]  = get_deriv(x[i-1:i+2], y[i-1:i+2], x[i], 1)
        
    return d

def calc_dd(x,y, order):
    dd     = np.zeros(x.shape)
    shift  = math.floor(order/2.0)
    print("shift ={}".format(shift))
    for i,_ in enumerate(x):
        l = i - shift
        u = i + shift + 1
        
        if(l < 0):
            u -= l
            l -= l
            
        if(u > x.shape[0]):
            l -= u - x.shape[0]
            u -= u - x.shape[0]

        # print("i -> {}; l -> {}; u -> {}".format(i,l,u))
            
        dd[i] = get_deriv(x[l:u], y[l:u], x[i], 2)
    return dd

x = np.random.rand(1000)*2*np.pi
x = np.sort(x)
y = np.sin(x)

dd =  calc_dd(x,y,5)

plt.plot(x,y, ".-", label="f(x)")
plt.plot(x,dd, label="f''(x) FD")
plt.plot(np.linspace(0,2*np.pi, 300) ,-np.sin(np.linspace(0,2*np.pi, 300)), ":", label="f'(x) analy")

plt.legend()
plt.show()


