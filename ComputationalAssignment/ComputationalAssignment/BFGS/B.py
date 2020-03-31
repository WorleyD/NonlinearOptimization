import numpy as np
import math

epsilon = 0.0000001

#helper function for that ugly sum to make the rest of the program nicer
def h(x):
	x = np.array(x).tolist()
	return sum([i**3*(x[i-1]-1)**2 for i in range(1,11)])

def norm(x):
	x = x.tolist()[0]
	return math.sqrt(sum([xi*xi for xi in x]))

def f(x):
	x = np.array(x)[0]

	return h(x)**3

# x is a vector with 10 elements
def gradient(x):
	x = x.tolist()[0]
	return np.asmatrix([h(x)**2*(2*j**3*(x[j-1]-1)) for j in range(1,11)])


def hessian(x):
	x = x.tolist()[0]
	hf = []
	for i in range(1,11):
		g_i = 6*h(x)*(2*i**3*(x[i-1]-1))
		rowi = []
		for j in range(1,11):
			xij = g_i * (2*j**3*(x[j-1]-1))
			if i == j:
				xij += 3*(h(x)**2)*(2*i**3)
			rowi.append(xij)
		hf.append(rowi)

	return np.asmatrix(hf)


xk = np.asmatrix(np.array([2.0 for i in range(10)]))
Dk = hessian(xk)
#Dk = np.array([[1,0],[0,1]])
iterations = 0

while True:
	prev = np.asmatrix(np.array([x for x in xk]))
	try:
		dk = np.matmul(np.linalg.inv(Dk), -1*np.transpose(gradient(xk)))
	except np.linalg.linalg.LinAlgError:
		pass


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
print(gradient([i for i in range(10)]))

print()


#Test if Hessian errors out
print(hessian([i for i in range(10)]))
'''