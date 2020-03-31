import numpy as np
from math import exp, sqrt

epsilon = 0.0000001


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
	x11 = sum([ 2*(2+2*i - exp(i*x1) - exp(i*x2)) * (-1*i*i*exp(i*x1)) + 2*(-1*i*exp(i*x1))**2 for i in range(1,11)])
	x12 = sum([ 2*(-1*i*exp(i*x1))*(-1*i*exp(i*x2)) for i in range(1,11)])
	x21 = x12
	x22 = sum([ 2*(2+2*i - exp(i*x1) - exp(i*x2))*(-1*i*i*exp(i*x2)) + 2*(-1*i*exp(i*x2))**2 for i in range(1,11)])
	return np.array([[x11,x12],[x21,x22]])


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


xk = np.array([5.0,5.0])
Dk = hessian(xk[0], xk[1])
iterations = 1

while True:
	prev = np.array([x for x in xk])
	dk = np.matmul(np.linalg.inv(Dk), -1*gradient(xk[0], xk[1]))

	a = backtrack(xk)
	sk = a*dk
	
	xk += sk
	yk = gradient(xk[0], xk[1]) - gradient(prev[0], prev[1])

	t1 = np.matmul(yk, np.transpose(yk))/(np.matmul(np.transpose(yk), sk))
	t2 = np.matmul(np.matmul(Dk, sk), np.matmul(np.transpose(sk), np.transpose(Dk)))

	Dk = Dk + t1 - t2/(np.matmul(np.transpose(sk), np.matmul(Dk, sk))) 


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