function c = center_of_mass(I)

[h,w] = size(I);
[y,x] = meshgrid(1:h, 1:w);

I = I / mean(I(:));

c(1) = mean(mean(I.*y));
c(2) = mean(mean(I.*x));

