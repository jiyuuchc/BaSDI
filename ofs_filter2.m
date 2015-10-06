function J = ofs_filter2(T, I, eps)
% J = ofs_filter2(T, I, eps)
% calculate offsetted 2D filter. Its equivalent to filter2(T + eps , I), but faster

J = filter2(T, I);
J = J + sum(I(:)) * eps;
