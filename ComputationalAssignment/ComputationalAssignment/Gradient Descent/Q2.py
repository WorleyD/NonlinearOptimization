import numpy as np
import math
epsilon = 0.000001


# Newtons returns min 0 at [1,1] after 5 iterations

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


def backtrack(x):
	g = gradient(x[0], x[1])	
	
	dx = -1*g

	a = 0.10	#initial backtrack amount
	l = 0.25	#amount to decrease by		
	
	t = 1
	while f(x[0], x[1]) + a*t*np.dot(g, dx) < f(x[0] + t*dx[0], x[1] + t*dx[1]):
		t *= l
		
	return t


xk = np.array([-2.0,2.0])
iterations = 0
while True:
	prev = np.array([x for x in xk])
	dk = -1*gradient(xk[0], xk[1])
	t = backtrack(xk)
	#print("t:", t)
	xk += t*dk
	#print(xk)
	if np.isnan(xk[0]):
		break
	if abs(norm(gradient(prev[0], prev[1])) - norm(gradient(xk[0], xk[1]))) < epsilon:
		print("Iterations: ", iterations)
		print("Minimizer: ", xk)
		print("Minimum: ", f(xk[0], xk[1]))
		break

	iterations += 1