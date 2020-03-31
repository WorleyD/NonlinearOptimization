import numpy as np
from math import sin,cos, sqrt

epsilon = 0.0000001


def norm(x):
	x = np.array(x)[0]
	return sqrt(sum([xi*xi for xi in x]))

def f(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]	
	return (x1*x1 + x2*x2 + x1*x2)**2 + sin(x1)*sin(x1) + cos(x2)*cos(x2)

def gradient(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]	
	x_1 = 2*(2*x1**3 + x2**3 + 3*x1*x1*x2 + 3*x1*x2*x2 + sin(x1)*cos(x1))
	x_2 = 2*(2*x2**3 + x1**3 + 3*x1*x1*x2 + 3*x1*x2*x2 - sin(x2)*cos(x2))
	return np.asmatrix([x_1,x_2])

def hessian(x):
	x = np.asarray(x)[0]
	x1,x2 = x[0], x[1]	
	x11 = 2*(6*x1*x1 + 6*x1*x2 + 3*x2*x2 + (cos(x1)*cos(x1) - sin(x1)*sin(x1)))
	x12 = 2*(3*x2*x2 + 3*x1*x1 + 6*x1*x2)
	x21 = x12
	x22 = 2*(6*x2*x2 + 6*x1*x2 + 3*x1*x1 + (sin(x2)*sin(x2) - cos(x2)*cos(x2)))
	return np.asmatrix([[x11,x12],[x21,x22]])

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
iterations = 1

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