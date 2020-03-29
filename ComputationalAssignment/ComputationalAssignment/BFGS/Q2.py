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



xk = np.array([0.0,0.0])
Dk = hessian(xk[0], xk[1])
iterations = 1
while True:
	prev = np.array([x for x in xk])
	xk -= np.matmul(np.linalg.inv(Dk), gradient(xk[0], xk[1]))
	dk = xk - prev
	yk = gradient(xk[0], xk[1]) - gradient(prev[0], prev[1])

	term1 = np.matmul(yk, np.transpose(yk))/(np.matmul(np.transpose(yk), dk))
	term2num = np.matmul(np.matmul(Dk, dk), np.transpose(dk))*Dk
	term2den = np.matmul(np.matmul(np.transpose(dk), Dk), dk)
	Dk = Dk + term1 + term2num/term2den

	if abs(norm(prev) - norm(xk)) < epsilon:
		print("Iterations: ", iterations)
		print("Minimizer: ", xk)
		print("Minimum: ", f(xk[0], xk[1]))
		break

	iterations += 1