function J = PSFBlur(I, psfFWHM)
% J = PSFBlur(I, psfFWHM)
% Gaussian filtering of a image.

s = psfFWHM / 2.355;
d = floor(psfFWHM * 3);
if (d < 5);
    d = 5;
end
h = fspecial('gaussian',[d,d],s);
J = imfilter(I,h, 'circular');
