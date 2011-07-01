%This script solves problems one and two from the pagerank lab. It
%implements all three methods of problem two.

load internet.dat;
internet(:,1:2) = internet(:,1:2) + 1;
A = spconvert(internet);

%This section does problem 1.
% spy(A)
% B = (A + A') > 0;
% L = spdiags(sum(B)',0,length(B),length(B))-B;
% eigs(L,5,'sm')

A = full(A(1:6000,1:6000));
%Atest = A(1:6000,1:6000);
s = sum(A,2);
e = ones(1,length(A));
sum(s == 0)
test = 0;

for i = 1:length(s)
    if s(i)>0
    else
        test = test +1;
        A(i,:) = e;
        %Atest(i,:) = e;
    end
end


K = (diag(1./sum(A,2))*A)';
d = .85;tic;
%The Algebraic Method
R = (eye(size(K))-d*K)\((1-d)*e'/length(e)); toc
tic;
%The eigenvector method
[V d] = eigs(d*K + (1-d)*ones(6000)/6000,1);
toc;
V = V/sum(V);
tic;
%The iterative method
Rinit = ones(6000,1)/6000;
Rold = Rinit;
convDist = 1;
while convDist > 2e-3
    Rnew = d*K*Rold + (1-d)*Rinit;
    convDist = norm(Rnew-Rold);
    Rold = Rnew;
end
toc;

max(Rnew)

max(V)

max(R)