import numpy as np
import math
epsilon = 0.000001

def norm(x):
	x = x.tolist()
	return math.sqrt(sum([xi*xi for xi in x]))

def f(x, y):
	return (1-x)**2 + 100*((y-x*x)**2)

def gradient(x,y):
	x1 = -2*(1-x) + 200*(y-x*x)*-2*x
	x2 = 200*(y-x*x)
	return np.array([x1,x2])

def hessian(x,y):
	x11 = 2- 400*y + 1200*x*x
	x12 = -400*x
	x21 = x12
	x22 = 200
	return np.array([[x11,x12],[x21,x22]])



xk = np.array([-2.0,-2.0])
iterations = 0
while True:
	prev = np.array([x for x in xk])
	dk = -1*np.matmul(np.linalg.inv(hessian(xk[0], xk[1])),gradient(xk[0], xk[1]))
	xk += dk
	if abs(norm(gradient(prev[0], prev[1])) - norm(gradient(xk[0], xk[1]))) < epsilon:
		print("Iterations: ", iterations)
		print("Minimizer: ", xk)
		print("Minimum: ", f(xk[0], xk[1]))
		break

	iterations += 1