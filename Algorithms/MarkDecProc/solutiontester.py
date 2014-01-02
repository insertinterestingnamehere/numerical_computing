import scipy as sp
import numpy as np
from scipy import stats as st
from scipy import linalg as la
import solutions as sol

data=sol.sim_data(sp.array([[108,200],[206,400],[74,140],[4,8]]),100000)
probs=sol.probabilities(data)
pulls=sol.numPulls(probs,1300)


if(sum(pulls)!=1300):
	print("Total pulls does not equal to M")
else:
	if (pulls[0]>320 and pulls[0]<380 and pulls[1]>130 and pulls[1]<190 and pulls[2]>270 and pulls[2]<330):
		print("Passed")
	else:
		print("Number of pulls are not in range")
