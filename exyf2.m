function e = exyf2(logtheta, o, max_shift)
%compute e(dx, dy) = P(dx,dy|theta,o)

[h,w] = size(logtheta);
e = zeros(max_shift * 2 + 1, max_shift * 2 + 1);
oi = ij_to_image(o - max_shift, h - max_shift * 2, w - max_shift * 2);

e = conv2(logtheta(h:-1:1,w:-1:1), oi, 'valid');

%convert back to linear scale
e(:) = e(:) - max(e(:));
e = exp(e(2 * max_shift + 1:-1:1, 2 * max_shift + 1:-1:1));

%normalize
s = sum(e(:));
e = e / s;
