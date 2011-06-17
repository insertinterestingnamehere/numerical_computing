function out = SimpsonsRule(fun,a,b)
out = ((b-a)/6)*[1 4 1]*fun([a;(b-a)/2;b]);