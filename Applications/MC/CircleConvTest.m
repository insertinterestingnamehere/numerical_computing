numTestPoints = 29;
testPoints = round(linspace(1000,100000,numTestPoints));
error = zeros(size(testPoints));
testRuns = 100;
for i = 1:numTestPoints
    for k = 1:testRuns
        numPoints = testPoints(i);
        points = rand(numPoints,2);
        points = 2*(points-.5);
        pointsNorm = hypot(points(:,1),points(:,2));
        InCircle = find(pointsNorm < 1);
        area(k) = 4*length(InCircle)/numPoints;
    end
    error(i) = mean(abs(area-pi));
end
size(error)
size(testPoints)

plot(log(testPoints),log(error))
estimate = [log(testPoints); ones(size(testPoints))]'\log(error');
convergenceRate = estimate(1)