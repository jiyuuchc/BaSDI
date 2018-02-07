function e = EXY(theta, O, max_shift)
%e_xy = E_XY(theta, O, max_shift)
%compute P(dx,dy|theta,O) for each individual frame as a function of drift d
%O: a cell array of 0-based coordinates for every frame
%theta: proportional to the real image, no need to be normalized

if (nargin < 3)
    max_shift = 20;
end

bg = 1; % allow a fixed false positive probability as a background signal. 

[h,w] = size(theta);
if (h <= max_shift * 2 + 1 || w <= max_shift * 2 + 1)
    error('Image size too small to allow shifting calculation');
end
eps = (max_shift * 2 + 1)^(-2);
% theta = theta / sum(theta(:)); % normalize

% initialize 
theta = theta + bg; % add a small chance for bg noise
logtheta = log(theta); % compute in log space to avoid overflow

parfor k = 1:length(O);
    o = O{k};
    if (length(o) > 0) % make sure it's not an empty frame (no molecule)
        
		% all the real work is done in exyf2
        e(:,:,k) = exyf2(logtheta, o, max_shift);
        
    else % for empty frame, simple assign a constant probability distribution
        
        e(:,:,k) = eps;

    end
end
