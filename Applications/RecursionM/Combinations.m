function out = Combinations(values,k)
%This functions outputs all possible combinations of k elements from the
%vector values.
if min(size(values)) ~= 1
    error('values must be a vector');
end
if k > length(values)
    error('k must be smaller than length(values)');
end
if (k <=0 || mod(k,1) ~= 0) 
    error('k must be a positive integer'); 
end

%Make input vectors column vectors
if size(values,2) > size(values,1)
    values = values';
end

out = []; 
n = length(values); 
if k == 1
    out = values; 
else
    
%This loop iterates through all of elements of the vector values that have at least k
%elements after them (inclusive). For each element it then calls
%Combinations(values(i+1:end),k-1), which returns combinations of size k-1
%for the elements succeeding the current element. This is so that we do not
%get repeats of combinations.
for i = 1:n-(k-1)
    % Calculate the number of possible combinations (to allow proper
    % concatenation in the recursive call.
    numCombs = factorial(n-i)/((factorial(k-1))*(factorial(n-i-(k-1))));
    %This is the recursive call.
    out = [out;[values(i)*ones(numCombs,1), Combinations(values(i+1:end),k-1)]];
end
end