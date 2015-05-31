function theta = update_theta(O,h,w,g)

theta = zeros(h,w);
[dh, dw, frames] = size(g);

maxshift = (dh - 1)/2;
[x,y] = meshgrid(-maxshift:maxshift,-maxshift:maxshift);
gn = g(:,:,1);
gn = gn / sum(gn(:));

cx = sum(sum(x.*gn));
cy = sum(sum(y.*gn));

for k = 1:frames;

    I = ij_to_image(O{k}, h, w);
    %theta = theta + imfilter(I, g(:,:,k),'conv');
    theta = theta + conv2(I, g(:,:,k),'same');

end

theta = circshift(theta, - round([cy, cx]));
