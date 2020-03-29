import numpy as np
import math

epsilon = 0.000001


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


xk = np.array([0.0]*10)
Dk = hessian(xk)
iterations = 1
singular = False
while True:
	prev = np.array([x for x in xk])
	try: 
		xk -= np.matmul(np.linalg.inv(Dk), gradient(xk))
	except np.linalg.linalg.LinAlgError:
		singular = True
	dk = xk - prev
	yk = gradient(xk) - gradient(prev)

	term1 = np.matmul(yk, np.transpose(yk))/(np.matmul(np.transpose(yk), dk))
	term2num = np.matmul(np.matmul(Dk, dk), np.transpose(dk))*Dk
	term2den = np.matmul(np.matmul(np.transpose(dk), Dk), dk)
	Dk = Dk + term1 + term2num/term2den

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