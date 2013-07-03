
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{graphicx}
\begin{document}
\title{Algorithms, Optimization, Optimization Package 2}

There are several optimization libraries for python. this lab explores the cvxopt package.
.

You can learn about more about cvxopt at  
%\url{http://abel.ee.ucla.edu/cvxopt/documentation/}.



\section{Linear Programs}

Cvxopt has linear program solver and can implement integer programming through the Gnu Linear Programming Kit, glpk.


Consider the following linear program. A shipping company makes money by charging a pickup fee and a delivery fee. Each




Consider the following transportation problem:
A piano company needs to transport thirteen pianos from their current locations from three supply centers to two demand centers. Transporting a piano from a supply center to a demand center incurs a cost, listed in the table below. The company want to minimize shipping costs for the pianos. How many pianos should each supply center send each demand center?

\begin{table}
Supply Center & Number of pianos available\\
1 & 7\\
2 & 2\\
3 & 4\\
\end{table}

\begin{table}
Demand Center & Number of pianos needed\\
4 & 5\\
5 & 8\\
\end{table}

\begin{table}
Supply Center & Demand Center & Cost of transportation & Number of pianos transported\\
1 & 4 & 4 & p\\
1 & 5 & 7 & q\\
2 & 4 & 6 & r\\
2 & 5 & 8 & s\\
3 & 4 & 8 & t\\
3 7 5 & 9 7 u\\
\end{table}

The variables $p,q,r,s,t,$ and $u$ must be nonnegative and satisfy the following three supply and two demand constraints:

\begin{align}
p +& q  &    &    &    &   =& 7\\
&  &    & r +& s  &    &   =& 2\\
&  &    &    &    & t +& u =& 4\\
p +&    & r +&    & t      =& 5\\
     q +&    & s +&    & u =& 8\\
\end{align}

The objective function is the number of pianos shipped from each location multiplied by the cost.

\begin{center}
$4p + 7q + 6r + 8s + 8t + 9u$
\end{center}

We can solve this program using cvxopt, by creating the matrices $A, b$ and $c$. 
Notice that all entries are floats and that $A$ consists of the columns of the constraints.

\text{
from cvxopt import matrix, solvers

>>> A = matrix([[1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0]])

>>> b = matrix([7.0, 2.0, 4.0, 5.0, 8.0])

>>> c = matrix([4.0, 7.0, 6.0, 8.0, 8.0, 9.0])

>>> sol=solvers.lp(c,A,b)   

\end{document}