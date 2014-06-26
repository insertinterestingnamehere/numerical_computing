import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Series Problem
s1 = pd.Series(-3, index=range(2, 11, 2))
s2 = pd.Series({'Bill':31, 'Sarah':28, 'Jane':34, 'Joe':26})

# Random Walk Problem
# five random walks of length 100 plotted together
N = 100
for i in xrange(5):
    s1 = np.zeros(N)
    s1[1:] = np.random.binomial(1, .5, size=N-1)*2-1
    s1 = pd.Series(s1)
    s1 = s1.cumsum()
    s1.plot()
plt.show()

# biased random walks
N = 100        #length of random walk
s1 = np.zeros(N)
s1[1:] = np.random.binomial(1, .51, size=(N-1,))*2-1 #coin flips
s1 = pd.Series(s1)
s1 = s1.cumsum()  #random walk
plt.subplot(311)
s1.plot()

N = 10000        #length of random walk
s1 = np.zeros(N)
s1[1:] = np.random.binomial(1, .51, size=(N-1,))*2-1 #coin flips
s1 = pd.Series(s1)
s1 = s1.cumsum()  #random walk
plt.subplot(312)
s1.plot()

N = 100000        #length of random walk
s1 = np.zeros(N)
s1[1:] = np.random.binomial(1, .51, size=(N-1,))*2-1 #coin flips
s1 = pd.Series(s1)
s1 = s1.cumsum()  #random walk
plt.subplot(313)
s1.plot()
  
plt.show()

# SQL SELECT problem
studentInfo[(studentInfo['Age']>19)&(studentInfo['Sex']=='M')][['ID', 'Name']]

# SQL JOIN problem
pd.merge(studentInfo[studentInfo['Sex']=='M'], otherInfo, on='ID')[['ID', 'Age', 'GPA']]

# final Crime Data problem
# load in the data
crimeDF = pd.read_csv("crime_data.txt", header=1, skiprows=0, index_col=0)
# create crime rate column
crimeDF['Crime Rate'] = crimeDF['Total']/crimeDF['Population']
# plot the crime rate as function of year
crimeDF.plot(y='Crime Rate')
# list 5 year with highest crime rate in descending order
crimeDF.sort(columns="Crime Rate", ascending=False).index[:5]
# calculate average total number of crimes, and average number of burglaries
avg = crimeDF.mean(axis=0)[['Total', 'Burglary']]
# find the years for total crime is below average, but burglary is above average
crimeDF[(crimeDF['Total']<avg['Total']) & (crimeDF['Burglary']>avg['Burglary'])].index
# plot murders as function of population
crimeDF.plot(x='Population', y='Murder')
# make histogram of Robbery and Burglary, plot side-by-side
crimeDF.hist(column=['Robbery', 'Burglary'])
# select Population, Violent, and Robbery columns for years in the 80s, save to csv file.
crimeDF.loc[1980:1989,['Population', 'Violent', 'Robbery']].to_csv("crime_subset.txt")

