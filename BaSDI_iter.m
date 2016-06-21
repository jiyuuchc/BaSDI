function [S,P] = BaSDI_iter(O, h, w, parameters, theta)

% Preprocessing
if (nargin < 5)
    theta = construct_palm(O, h, w);
end

if (nargin < 4)
    parameters = struct();
end

if ~ isfield(parameters, 'p')
    parameters.p = 0.1;
end
if ~ isfield(parameters, 'eps')
    parameters.eps = 0;
end
if ~ isfield(parameters, 'max_shift')
    parameters.max_shift = 30;
end
if ~ isfield(parameters, 'smooth')
    parameters.smooth = 2;
end

OC = remove_border(O, h, w, parameters.max_shift);

%E step
display('E step');
theta2 = PSFBlur(theta,parameters.smooth); 
e = EXY(theta2, OC, parameters.max_shift);
[g, g_s] = for_back(e, parameters.p, parameters.eps);

%M step
display('M step');
theta = update_theta(O, h, w, g);

%figure(1); plot(processing_result(g));
%figure(2);imagesc(theta); colormap('gray');
%input('press enter');

S.theta = theta;
S.e = e;
S.g = g;
S.dim = [h,w];

P = parameters;
