import numpy as np
import math

epsilon = 0.0000001

def norm(x):
	x = x.tolist()
	return math.sqrt(sum([xi*xi for xi in x]))

def f(x1,x2):
	return (-13 + x1 - 2*x2 + 5*x2*x2 - x2**3)**2 + (-29 + x1 - 14*x2 + x2*x2 + x2**3)**2


def gradient(x1,x2):
	x_1 = -84 + 4*x1 - 32*x2 + 12*x2*x2
	x_2 = 864 + 24*x2 -32*x1 - 240*x2*x2 + 24*x1*x2 +8*x2**3 - 40*x2**4 + 12*x2**5
	return np.array([x_1, x_2])

def hessian(x1,x2):
	x_11 = 4
	x_12 = -32 + 24*x2
	x_21 = x_12
	x_22 = 24*x1 + 60*x2**4 - 160*x2**3 + 24*x2**2 - 480*x2 + 24
	return np.array([[x_11, x_12], [x_21, x_22]])

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
a = np.array([[1.,2.],[3.,4.]])
ainv = np.linalg.inv(a)

print(ainv)

#Test if it errors out
print(gradient(1,2))

print()

#Test if Hessian errors out
print(hessian(1,2))
'''