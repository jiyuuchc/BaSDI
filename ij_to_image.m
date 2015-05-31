function I = ij_to_image(ij, h, w)
% I = ij_to_image(ij, h, w)
% convert a set of coordinates to an binary image

I = zeros(h,w);
if (size(ij,1) > 0);
    idx = ij(:,2) * h + ij(:,1) + 1;
    I(idx) = 1;
end
