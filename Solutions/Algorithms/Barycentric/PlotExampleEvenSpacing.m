ipoints = linspace(-1,1,15);
runge =@(x) 1./(1 + 25*x.^2);
xin = linspace(-1,1,1000);
plot(xin,LInterp1(xin,ipoints,runge(ipoints)),'r');
hold on;
plot(xin,runge(xin));