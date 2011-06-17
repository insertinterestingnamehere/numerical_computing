function seeds = complex_basins(f, df, roots, resolution)
close all;
%Given an function handle, find the basins of attraction for newtons method
%on the interval [-10,10]

    %First we code up newtons method
    function root = newt(f, df,x0)
    %This is going to be real simple, we'll do 20 iterations just to be
    %safe, we don't really care to look up things that attract strangely
        numiter = 20;
        j = 1;
        x = x0;
        while j < numiter
            x = x - f(x)/df(x);
            j = j+1;
        end
        root = x;
    end


%Generate the seeds.  1000 should be enough

%seeds = [linspace(-10,10,resolution)' zeros(resolution,1)];
seeds = zeros(resolution, resolution, 2);
b = sqrt(-1);
seeds(:,:,1) = ones(resolution,1)*linspace(-10,10,resolution);
seeds(:,:,1) = flipud(seeds(:,:,1) + b.*seeds(:,:,1)');


for i = 1:resolution
    i
    for k = 1:resolution
        seeds(i,k,2) = newt(f,df,seeds(i,k,1));
    end
end

%plot the function and the basins
hold on
%plot(seeds(:,1),f(seeds(:,1)));

colors = linspace(0,1,length(roots));
for i = 1:length(roots)
    mask = abs(seeds(:,:,2) - roots(i)) < 0.001;
    plot(seeds(mask),'color',[0 colors(i) 0],'marker','.','LineStyle','none');
end
hold off
end


