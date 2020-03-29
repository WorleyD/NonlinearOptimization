import numpy as np
import math

epsilon = 0.000001
singular = False

def norm(x):
	x = x.tolist()
	return math.sqrt(sum([xi*xi for xi in x]))

def f(x):
	return (sum([i**3*(x[i-1]-1)**2 for i in range(1,11)]))**3

# x is a vector with 10 elements
def gradient(x):
	return None if len(x) != 10 else np.array([(3*(sum([(i**3)*(x[i-1]-1)**2 for i in range(1,11)]))**2)*(2*i**3)*(x[i-1]-1) for i in range(1,11)])


def hessian(x):
	hf = []
	for i in range(1,11):
		g_i = sum([i*i*i*(x[i-1]-1)**2])*(2*i*i*(x[i-1]-1))
		rowi = []
		for j in range(1,11):
			xij = g_i * (2*i**3*(x[j-1]-1))
			if i == j:
				xij += 3*(sum([i**3*(x[i-1]-1)]))*2*i**3
			rowi.append(xij)
		hf.append(rowi)

	return np.array(hf)


xk = np.array([-1.0]*10)
iterations = 1
while True:
	prev = np.array([x for x in xk])
	try:
		dk = -1*np.matmul(np.linalg.inv(hessian(xk)),gradient(xk))
		xk += dk
	except np.linalg.linalg.LinAlgError:
		singular = True
	if abs(norm(prev) - norm(xk)) < epsilon or singular:
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