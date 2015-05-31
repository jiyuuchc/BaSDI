function O = gen_palm_data(I, n, d)

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
