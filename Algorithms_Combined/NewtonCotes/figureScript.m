%This script generates the figures for the Newton Cotes Integration Section


%Generates the first figure, simple trapezoid rule
verts = [0,0;0,1;1,0];
faces = [1 2 3];
p = patch('Faces',faces,'Vertices',verts);
set(p,'FaceColor','flat','FaceVertexCData',[.7;.7;.7]);
hold on;
x = linspace(0,1,100);
plot(x,1-x.^4,'r','LineWidth',2);
plot(x,1-x,'k','LineWidth',2);
axis([-.1 1.1 -.5 1.5]);
grid on;


%Generates the second figure, composite rule
figure;
test = linspace(0,1,5)';
verts = [test zeros(size(test)); test 1-test.^4];
faces = [1 2 7 6;2 3 8 7; 3 4 9 8;4 5 10 9];
p = patch('Faces',faces,'Vertices',verts);
set(p,'FaceColor','flat','FaceVertexCData',.7*ones(10,1));
hold on;
plot(test,1-test.^4,'k','LineWidth',2);
plot(x,1-x.^4,'r','LineWidth',2);
axis([-.1 1.1 -.5 1.5]);
grid on;