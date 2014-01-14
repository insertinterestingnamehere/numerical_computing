import numpy as np
import solutions as sol

prob11=np.array([[1.54643957],[ 1.20887997],[ 1.89941363]])
prob12=10
prob21=np.array([[ 5.00e+00],[ 2.00e+00],[ 5.55e-08],[ 2.00e+00],[ 1.05e-08],[ 4.00e+00]])
prob22=86

prob31=np.array([[-1.50],[ 1.00],[-.5]])
prob32=-2.5

prob41=np.array([[ 1.41e-08],[ 6.76e-08],[ 7.50e+01],[ 9.00e+01],[ 1.28e-07],[ 2.52e-07],[ 1.40e+02],[ 4.18e-07],[ 5.52e-06],[ 1.04e-08],[ 8.94e-09],[ 6.00e+01],[ 1.23e-07],[ 1.54e+02],[ 5.80e+01],[ 3.16e-08],[ 3.58e-08],[ 9.80e+01],[ 1.63e-08],[ 9.12e-09],[ 1.13e+02]])
prob42=322514998.983

x,y=sol.Problem1()

if(np.allclose(prob11,np.array(x)) and np.allclose(prob12,y)):
	print("Problem1 Passed")
else:
	print("Problem1 Falied")
	print("Your answer:")
	print(np.array(x))
	print(y)
	print("Correct answer:")
	print(prob11)
	print(prob12)

x,y=sol.Problem2()

if(np.allclose(prob21,np.array(x)) and np.allclose(prob22,y)):
	print("Problem2 Passed")
else:
	print("Problem2 Falied")
	print("Your answer:")
	print(x)
	print(y)
	print("Correct answer:")
	print(prob21)
	print(prob22)

x,y=sol.Problem3()

if(np.allclose(prob31,np.array(x)) and np.allclose(prob32,y)):
	print("Problem3 Passed")
else:
	print("Problem3 Falied")
	print("Your answer:")
	print(x)
	print(y)
	print("Correct answer:")
	print(prob31)
	print(prob32)

x,y=sol.Problem4()

if(np.allclose(prob41,np.array(x)) and np.allclose(prob42,y)):
	print("Problem4 Passed")
else:
	print("Problem4 Falied")
	print("Your answer:")
	print(x)
	print(y)
	print("Correct answer:")
	print(prob41)
	print(prob42)
