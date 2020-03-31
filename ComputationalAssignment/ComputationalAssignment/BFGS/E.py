import numpy as np
from math import exp, sqrt

epsilon = 0.0000001

def norm(x):
	x = np.array(x)[0]
	return sqrt(sum([xi*xi for xi in x]))

def f(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]
	return sum([(2+2*i - exp(i*x1) - exp(i*x2))**2 for i in range(1,11)])

def gradient(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]	
	x_1 = sum([2*( (2+2*i - exp(i*x1) - exp(i*x2))*( -1*i*exp(i*x1) ) ) for i in range(1,11)])
	x_2 = sum([2*( (2+2*i - exp(i*x1) - exp(i*x2))*( -1*i*exp(i*x2) ) ) for i in range(1,11)])
	return np.asmatrix([x_1, x_2])

def hessian(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]	
	x11 = sum([ 2*(2+2*i - exp(i*x1) - exp(i*x2)) * (-1*i*i*exp(i*x1)) + 2*(-1*i*exp(i*x1))**2 for i in range(1,11)])
	x12 = sum([ 2*(-1*i*exp(i*x1))*(-1*i*exp(i*x2)) for i in range(1,11)])
	x21 = x12
	x22 = sum([ 2*(2+2*i - exp(i*x1) - exp(i*x2))*(-1*i*i*exp(i*x2)) + 2*(-1*i*exp(i*x2))**2 for i in range(1,11)])
	return np.asmatrix([[x11,x12],[x21,x22]])



xk = np.asmatrix(np.array([1.0,1.0]))
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
#Test if it errors out
print(gradient(1,2))

print()


#Test if Hessian errors out
print(hessian(1,2))
'''