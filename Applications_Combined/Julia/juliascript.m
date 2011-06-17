niter=15;
m=500;
cx=0;
cy=0;
l=1.5;
x=linspace(cx-l,cx+l,m);
y=linspace(cy-l,cy+l,m);
[X,Y]=meshgrid(x,y);
%c= -.745429;

c = -0.8 + 0.125*i;
Z=X+i*Y;
for k=1:niter;
    Z=Z.^2+c;
    W=exp(-abs(Z));
end

%colormap prism(256)
%pcolor(W);
colormap jet;
imagesc(W)
%shading flat;
axis('square','equal','off');
