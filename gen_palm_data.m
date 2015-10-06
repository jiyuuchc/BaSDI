function O = gen_palm_data(I, n, d)
% Simple simulation to generate a localization imaging dataset
% O = gen_palm_data(I, n, d)
% Inputs:
% I: ground truth image.
% n: average number of molecules detected in each frame
% d: a drift trace (N x 2 array). The length of d also determined the number  of image frames generated
% Output:
% O: a cell array representing the localization dataset.

[h,w] = size(I);

d = round(d);
ni = double(I(:));
ni = ni / sum(ni) * n;

for i = 1:length(d);
    r = rand(length(ni),1);
    img = circshift(reshape(r < ni,h,w), d(i,:));
    
    idx = find(img) - 1;

    cols = floor(idx / h);
    o = idx - cols * h;
    o(:,2) = cols;
    O{i} = o;
end
