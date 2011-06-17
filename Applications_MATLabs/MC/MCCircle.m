numPoints = 500;
points = rand(numPoints,2);
points = 2*(points-.5);
pointsNorm = hypot(points(:,1),points(:,2));
InCircle = find(pointsNorm < 1);
OutCircle = find(pointsNorm > 1);
plot(points(InCircle,1),points(InCircle,2),'r.');
hold on;
plot(points(OutCircle,1),points(OutCircle,2),'b.');

%Plots a Circle
theta = linspace(0,2*pi,50);
plot(cos(theta),sin(theta),'k');

axis off;
axis equal;

area = 4*length(InCircle)/numPoints;