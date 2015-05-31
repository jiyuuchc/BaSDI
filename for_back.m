function [g, g_s] = for_back(e_xy, p, eps)
% function [g, g_s] = for_back(e, p)
% forward_backward alogorithm for computing marginal probability
% of Markovian process

if (nargin < 2)
    p = 0.2; % default variance of the gaussian filter 
end

% h,w is the size of the shifting matrix
% f is the number of frames
[h,w,f] = size(e_xy);

if (nargin < 3)
    eps = 0;
end

%default transition matrix
T = fspecial('gaussian', [3, 3], sqrt(p)); 
% T = [p^2, p, p^2;
%     p, 1, p;
%     p^2, p, p^2];
% T = T / sum(T(:)); %normalize

%forward computation
a(:,:,1) = e_xy(:,:,1);
a_s = zeros(1,f);

for i = 2:f;
    e_i = e_xy(:,:,i);
    a_t = ofs_filter2(T, a(:,:,i-1), eps) .* e_i;
    a_m = max(a_t(:));
    a(:,:,i) = a_t / a_m;
    a_s(i) = a_s(i - 1) + log(a_m);
end

%backward computation
b(:,:,f) = zeros(w,h) + 1;
b_s = zeros(1,f);

for i = f-1:-1:1
    e_i = e_xy(:,:,i);
    b_t = ofs_filter2(T,b(:,:,i+1).*e_i, eps);
    b_m = max(b_t(:));
    b(:,:,i) = b_t / b_m;
    b_s(i) = b_s(i+1) + log(b_m);
end

%calculate the probability
g = a.*b;
for i = 1:f
    gk = g(:,:,i);
    g(:,:,i) = gk / sum(gk(:));
end
g_s = a_s + b_s;
