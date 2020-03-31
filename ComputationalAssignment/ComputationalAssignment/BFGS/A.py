import numpy as np
import math

epsilon = 0.00000001

def norm(x):
	x = np.asarray(x)
	x = x[0].tolist()
	return math.sqrt(sum([xi*xi for xi in x]))

def f(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]
	return (-13 + x1 - 2*x2 + 5*x2*x2 - x2**3)**2 + (-29 + x1 - 14*x2 + x2*x2 + x2**3)**2


def gradient(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]
	x_1 = -84 + 4*x1 - 32*x2 + 12*x2*x2
	x_2 = 864 + 24*x2 -32*x1 - 240*x2*x2 + 24*x1*x2 +8*x2**3 - 40*x2**4 + 12*x2**5
	return np.asmatrix([x_1, x_2])

def hessian(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]	
	x_11 = 4
	x_12 = -32 + 24*x2
	x_21 = x_12
	x_22 = 24*x1 + 60*x2**4 - 160*x2**3 + 24*x2**2 - 480*x2 + 24
	return np.asmatrix([[x_11, x_12], [x_21, x_22]])

def backtrack(x):
	a = 0.5
	p =0.75
	t = 1

	g = gradient(x[0],x[1])
	newx = x - t*g
	while f(newx[0], newx[1]) > f(x[0], x[1]) - t*a*(norm(g))**2:
		t *= p
		newx = x - t*g
	return p 


xk = np.asmatrix(np.array([10.0,10.0]))
Dk = hessian(xk)
#Dk = np.array([[1,0],[0,1]])
iterations = 0

while True:
	prev = np.asmatrix(np.array([x for x in xk]))
	dk = np.matmul(np.linalg.inv(Dk), -1*np.transpose(gradient(xk)))
	
	xk =  xk + np.asarray(np.transpose(dk))[0]
	
	
	yk = gradient(xk) - gradient(prev)


	t1 = np.matmul(np.transpose(yk), np.asmatrix(yk))
	t2 = np.matmul(yk, dk)
	
	t3 = np.matmul(Dk, np.matmul(dk,np.matmul(np.transpose(dk), np.transpose(Dk))))
	t4 = np.matmul(np.transpose(dk), np.matmul(Dk, dk))

	Dk = Dk + t1/t2 - t3/t4 


	if abs(norm(prev) - norm(xk)) < epsilon:
		print("Iterations: ", iterations)
		print("Minimizer: ", xk)
		print("Minimum: ", f(xk))
		break

	iterations += 1

'''
a = np.array([[1.,2.],[3.,4.]])
ainv = np.linalg.inv(a)

print(ainv)

#Test if it errors out
print(gradient(1,2))

print()

#Test if Hessian errors out
print(hessian(1,2))
'''