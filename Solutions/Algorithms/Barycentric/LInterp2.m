function out = LInterp2(xin,ipoints,fpoints)
%Barycentric Lagrange Interpolation
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

numer = 0;
denom = 0;

for k = 1:length(ipoints)
    numer = numer + w(k)*fpoints(k)./(xin-ipoints(k));
    denom = denom + w(k)./(xin-ipoints(k));
end
out = numer./denom;