function out = LInterp3(xin,ipoints,fpoints)
%An attempt to adjust the initial method using some ideas from the
%barycentric one...not very successfully
out = zeros(size(xin));

D = max(ipoints)-min(ipoints);
%D = 2;
C = D/4;
w = zeros(size(ipoints));
%Calculate Barycentric Weights
shuffle = randperm(length(ipoints)-1);
for k = 1:length(ipoints)
    test = (ipoints(k)-ipoints)/C;
    test(k) = [];
    test = test(shuffle);
    w(k) = 1/prod(test);
end

for j = 1:length(ipoints)%evaluate each basis polynomial
    numer = ones(size(xin));
    for k = 1:length(ipoints)
        if k == j
        else
            numer = numer.*(xin-ipoints(k));
        end
    end
    out = out + fpoints(j)*numer*w(j);
end
