function J = PSFBlur(I, psfFWHM)

s = psfFWHM / 2.355;
d = floor(psfFWHM * 3);
if (d < 5);
    d = 5;
end
h = fspecial('gaussian',[d,d],s);
J = imfilter(I,h, 'circular');
