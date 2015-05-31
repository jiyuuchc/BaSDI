function J = ofs_filter2(T, I, eps)
% J = ofs_filter2(T, I, eps)
% calculate offsetted 2d filter: 
% J = filter2(T + eps , I);

J = filter2(T, I);
J = J + sum(I(:)) * eps;
