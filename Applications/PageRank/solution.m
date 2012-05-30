load internet.dat;
internet(:,1:2) = internet(:,1:2) + 1;
A = spconvert(internet);
% spy(A)
% B = (A + A') > 0;
% L = spdiags(sum(B)',0,length(B),length(B))-B;
% eigs(L,5,'sm')

A = full(A(1:6000,1:6000));
s = sum(A');
e = ones(1,length(A));
for i = 1:length(s)
    if s(i)>0
    else
        A(i,:) = e;
        i
    end
end
nnn = 5


K = (diag(1./sum(A'))*A)';
d = .85;tic;
R = (eye(size(K))-d*K)\((1-d)*e'/length(e)); toc
max(R)