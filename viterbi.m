function v = viterbi(e, p, eps)
% function v = viterbi(e)
% return most likely state sequence based on viterbi algorithm

if (nargin < 2)
    p = 0.1;
end

[dh, dw, f] = size(e);

if (nargin < 3)
    eps = 0;
end

%default transition matrix
T = [p^2, p, p^2, p, 1, p, p^2, p, p^2]';
T = T / sum(T) - eps; %normalize
[x, y] = meshgrid(-1:1, -1:1);
s = x(:) * (dh + 2) + y(:); %shifting offset

%forward
h = zeros(dh + 2, dw + 2, f);
vn = zeros(dh + 2, dw + 2);
vn(2:dh+1, 2:dw+1) = e(:,:,1);
vp = vn;
[m, idx2] = max(vp(:));
m = m * eps;

for i = 2:f
    for x = 2:dw+1;
        for y = 2:dh+1
            sn = s + (x - 1) * (dh + 2) + y;
            [mp, idx] = max(vp(sn) .* T);
            idx = sn(idx);
            if (mp < m )
                mp = m;
                idx = idx2;
            end    
            vn(y, x) = mp * e(y-1, x-1, i);
            h(y, x, i) = idx;
        end
    end
    vp = vn;
    vp = vp / max(vp(:));
    [m, idx2] = max(vp(:));
    m = m * eps;
end

%backward
v = zeros(f,2);
[temp, idx] = max(vp(:));
idx = idx - 1;
v(f,2) = floor(idx / (dh + 2));
v(f,1) = idx - v(f,2) * (dh + 2);

for i = f:-1:2
    hp = h(:,:,i);
    idx = hp(idx + 1);
    idx = idx - 1;
    v(i-1,2) = floor(idx / (dh + 2));
    v(i-1,1) = idx - v(i-1,2) * (dh + 2);
end

v(:,1) = v(:,1) - (dh + 1)/2;
v(:,2) = v(:,2) - (dw + 1)/2;
v = -v;

