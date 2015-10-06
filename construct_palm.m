function I = construct_palm(O, h, w, d)
% Output a reconstructed super-resolution image based on input data and drift
% I = construct_palm(O, h, w, d)
% Inputs:
% O: localization dataset. A cell array with N elements. N is the number of image frames.
% h: hight of the image
% w: width of the image
% d: drift trace. Nx2 array
% Output:
% I: reconstructed super-resolution image. An 2D array (h x w).

L = length(O);

if (nargin < 4)
    % use a efficient algorithm for no-drift PALM construction
    o = cat(1, O{:});
    idx = o(:,2) * h + o(:,1) + 1;
    [I,bins] = hist(idx, 1: h*w);

    I = reshape(I, h, w);

else 

    I = zeros(h, w);

    for i = 1:L;
        J = ij_to_image(O{i}, h, w);
        I = I + shift_image(J, -d(i, :));
    end
    
end
