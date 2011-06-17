f = @(x) atan(2*(x - .5));
xplot = linspace(-1,2,100);
x1 = .6;
x2 = 1.1;
g = @(x) (f(x1)-f(x2))*(x-x1)/(x1-x2) + f(x1);
x3 = -f(x1)*(x1-x2)/(f(x1)-f(x2))+x1;
close all;
plot(xplot,f(xplot));
hold on;
plot(xplot,g(xplot),'r');
plot([x1 x2 x3], [g([x1 x2]) 0],'ro')
plot(xplot,zeros(size(xplot)),'k')
