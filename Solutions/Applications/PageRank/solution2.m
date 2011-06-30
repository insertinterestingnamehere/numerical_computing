function out = solution2(tol,n)
tic;
load internet.dat;
internet(:,1:2) = internet(:,1:2) + 1;
A = spconvert(internet);
toc;
n = min(n,length(A));
% spy(A)
% B = (A + A') > 0;
% L = spdiags(sum(B)',0,length(B),length(B))-B;
% eigs(L,5,'sm')
size(A)

%n = 100000;
Atest = A(1:n,1:n);
s = sum(Atest,2);
inDiag = 1./s;
sinks = find(s == 0);
inDiag(sinks) = 0;
diagTest = spdiags(inDiag,0,n,n);
KTest = (diagTest*Atest)';




d = .85;

%Testing Sparse iterative methods
tic;
Rinit = ones(n,1)/n;
Rold2 = Rinit;
convDist = 1;
tic;
while convDist > tol
    Rnew2 = d*KTest*Rold2 + (1-d)*Rinit + d*Rinit*sum(Rold2(sinks));
    convDist = norm(Rnew2-Rold2);
    Rold2 = Rnew2;
end
toc;
Rnew2(1)
Rnew2(1964)
Rnew2(15587)
out = max(Rnew2);
out = [out find(Rnew2 == out)];