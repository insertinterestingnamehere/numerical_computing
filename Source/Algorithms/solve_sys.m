n = 3000;
A = rand(n);
b = rand(n,1);

tic;
A\b;
toc;

tic;
inv(A)*b;
toc;