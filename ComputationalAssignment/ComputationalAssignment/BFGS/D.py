import numpy as np
import math

A = np.asmatrix([
[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
[-1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2 ],
])

epsilon = 0.0000001

def norm(x):
	x = x.tolist()[0]
	return math.sqrt(sum([xi*xi for xi in x]))

def f(x):
	x = np.transpose(x)
	xt = np.transpose(np.array(x))
	res = np.matmul(xt,A)
	return np.array(np.matmul(res, x) - 2*x[0])[0][0]
# x: a 20 element vector
def gradient(x):
	x_p = np.matmul(2*A, np.transpose(x))
	x_p[0] -= 2
	return np.transpose(x_p)

# x: a 20 element vector
def hessian(x):
	return 2*A

def backtrack(x):
	a = 0.5
	p =0.75
	t = 1

	g = gradient(x)
	newx = x - t*g
	while f(newx[0], newx[1]) > f(x[0], x[1]) - t*a*(norm(g))**2:
		t *= p
		newx = x - t*g
	return p 



xk = np.asmatrix(np.array([10.0 for i in range(20)]))
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

# Test gradient
print(gradient([i for i in range(20)]))

print()
# Test Hessian
print(hessian([i for i in range(20)]))
'''