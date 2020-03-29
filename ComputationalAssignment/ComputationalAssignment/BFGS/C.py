import numpy as np
from math import sin,cos, sqrt

epsilon = 0.000001


def norm(x):
	x = x.tolist()
	return sqrt(sum([xi*xi for xi in x]))

def f(x1,x2):
	return (x1*x1 + x2*x2 + x1*x2)**2 + sin(x1)*sin(x1) + cos(x2)*cos(x2)

def gradient(x1,x2):
	x_1 = 2*(2*x1**3 + x2**3 + 3*x1*x1*x2 + 3*x1*x2*x2 + sin(x1)*cos(x1))
	x_2 = 2*(2*x2**3 + x1**3 + 3*x1*x1*x2 + 3*x1*x2*x2 + sin(x2)*cos(x2))
	return np.array([x_1,x_2])

def hessian(x1,x2):
	x11 = 2*(6*x1*x1 + 6*x1*x2 + 3*x2*x2 + (cos(x1)*cos(x1) - sin(x1)*sin(x1)))
	x12 = 2*(3*x2*x2 + 3*x1*x1 + 6*x1*x2)
	x21 = x12
	x22 = 2*(6*x2*x2 + 6*x1*x2 + 3*x1*x1 + (cos(x2)*cos(x2) - sin(x2)*sin(x2)))
	return np.array([[x11,x12],[x21,x22]])


xk = np.array([1.0,1.0])
Dk = hessian(xk[0], xk[1])
iterations = 1
while True:
	prev = np.array([x for x in xk])

	xk -= np.matmul(np.linalg.inv(Dk), gradient(xk[0], xk[1]))
	
	dk = np.array([xk[i] - prev[i] for i in range(len(xk))])
	
	t1 = gradient(xk[0], xk[1])
	t2 = gradient(prev[0], prev[1])
	yk = t1 - t2 

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