function out = NewtonInterp(xin,ipoints,fpoints)
%Newton Interpolation
out = zeros(size(xin));

for j = 1:length(ipoints)%evaluate each basis polynomial
    denom = 1;
    numer = ones(size(xin));
    for k = 1:length(ipoints)
        if k == j
        else
            numer = numer.*(xin-ipoints(k));
            denom = denom*(ipoints(j)-ipoints(k));
        end
    end
    out = out + fpoints(j)*numer/denom;
end

