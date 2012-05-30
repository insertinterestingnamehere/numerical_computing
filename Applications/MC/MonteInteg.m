function out = MonteInteg(fun,n,numPoints)
randVec = (1/.51)*(rand(numPoints,n)-.49);
fVals = zeros(numPoints,1);
for i = 1:numPoints
    fVals(i) = fun(randVec(i,:));
end
out = mean(fVals)*2^n;