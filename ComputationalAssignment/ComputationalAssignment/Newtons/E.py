import numpy as np
from math import exp, sqrt

epsilon = 0.000001


def norm(x):
	x = x.tolist()
	return sqrt(sum([xi*xi for xi in x]))

def f(x1,x2):
	return sum([(2+2*i - exp(i*x1) - exp(i*x2))**2 for i in range(1,11)])


def gradient(x1,x2):
	x_1 = sum([2*( (2+2*i - exp(i*x1) - exp(i*x2))*( -1*i*exp(i*x1) ) ) for i in range(1,11)])
	x_2 = sum([2*( (2+2*i - exp(i*x1) - exp(i*x2))*( -1*i*exp(i*x2) ) ) for i in range(1,11)])
	return np.array([x_1, x_2])

def hessian(x1,x2):
	x11 = sum([ 2*(2+2*i - exp(i*x1) - exp(i*x2)) * (-1*i*i*exp(i*x1)) + 2*(-1*i*exp(i*x1)**2) for i in range(1,11)])
	x12 = sum([ 2*(-1*i*exp(i*x1))*(-1*i*exp(i*x2)) for i in range(1,11)])
	x21 = x12
	x22 = sum([ 2*(2+2*i - exp(i*x1) - exp(i*x2))*(-1*i*i*exp(i*x2)) + 2*(-1*i*exp(i*x2)**2) for i in range(1,11)])
	return np.array([[x11,x12],[x21,x22]])


xk = np.array([10.0,10.0])
iterations = 1
while True:
	prev = np.array([x for x in xk])
	dk = -1*np.matmul(np.linalg.inv(hessian(xk[0], xk[1])),gradient(xk[0], xk[1]))
	xk += dk
	if abs(norm(prev) - norm(xk)) < epsilon:
		print("Iterations: ", iterations)
		print("Minimizer: ", xk)
		print("Minimum: ", f(xk[0], xk[1]))
		break

	iterations += 1

'''
#Test if it errors out
print(gradient(1,2))

print()


#Test if Hessian errors out
print(hessian(1,2))
'''