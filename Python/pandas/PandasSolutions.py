import pandas as pd
import numpy as np

# Series Problem
s1 = pd.Series(-3, index=range(2, 11, 2))
s2 = pd.Series({'Bill':31, 'Sarah':28, 'Jane':34, 'Joe':26})

# SQL SELECT problem
studentInfo[(studentInfo['Age']>19)&(studentInfo['Sex']=='M')][['ID', 'Name']]

# SQL JOIN problem
pd.merge(studentInfo[studentInfo['Sex']=='M'], otherInfo, on='ID')[['ID', 'Age', 'GPA']]

