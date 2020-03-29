import numpy as np
from math import exp, sqrt

epsilon = 0.000001


def norm(x):
	x = x.tolist()
	return sqrt(sum([xi*xi for xi in x]))

def f(x1,x2):
	return sum([2+2*i - exp(i*x1) - exp(i*x2) for i in range(1,11)])**2


def gradient(x1,x2):
	x_1 = sum([(-4*i-4*i*i)*exp(i*x1) + i*exp(2*i*x1) - i*exp(i*(x1+x2)) for i in range(1,11)])
	x_2 = sum([(-4*i-4*i*i)*exp(i*x2) + i*exp(2*i*x2) - i*exp(i*(x1+x2)) for i in range(1,11)])
	return np.array([x_1, x_2])

def hessian(x1,x2):
	x11 = sum([(-4*i*i -4*i*i*i)*exp(i*x1) + 2*i*i*exp(2*1*x1) - i*exp(i*(x1+x2)) for i in range(1,11)])
	x12 = sum([-i*i*exp(i*(x1+x2)) for i in range(1,11)])
	x21 = x12
	x22 =sum([(-4*i*i -4*i*i*i)*exp(i*x2) + 2*i*i*exp(2*1*x2) - i*exp(i*(x1+x2)) for i in range(1,11)])
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

'''
#Test if it errors out
print(gradient(1,2))

print()


#Test if Hessian errors out
print(hessian(1,2))
'''