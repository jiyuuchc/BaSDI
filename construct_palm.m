function I = construct_palm(O, h, w, d)

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
