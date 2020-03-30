import numpy as np
from math import sin,cos, sqrt

epsilon = 0.0000001


def norm(x):
	x = x.tolist()
	return sqrt(sum([xi*xi for xi in x]))

def f(x1,x2):
	return (x1*x1 + x2*x2 + x1*x2)**2 + sin(x1)*sin(x1) + cos(x2)*cos(x2)

def gradient(x1,x2):
	x_1 = 2*(2*x1**3 + x2**3 + 3*x1*x1*x2 + 3*x1*x2*x2 + sin(x1)*cos(x1))
	x_2 = 2*(2*x2**3 + x1**3 + 3*x1*x1*x2 + 3*x1*x2*x2 - sin(x2)*cos(x2))
	return np.array([x_1,x_2])

def hessian(x1,x2):
	x11 = 2*(6*x1*x1 + 6*x1*x2 + 3*x2*x2 + (cos(x1)*cos(x1) - sin(x1)*sin(x1)))
	x12 = 2*(3*x2*x2 + 3*x1*x1 + 6*x1*x2)
	x21 = x12
	x22 = 2*(6*x2*x2 + 6*x1*x2 + 3*x1*x1 + (sin(x2)*sin(x2) - cos(x2)*cos(x2)))
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

print(f(-0.15, 0.69))
'''

#Test if it errors out
print(gradient(1,2))

print()


#Test if Hessian errors out
print(hessian(1,2))
'''