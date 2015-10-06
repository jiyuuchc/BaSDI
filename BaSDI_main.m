function S = BaSDI_main(O, h, w)

% Change these to suit you need
% ---------------------------------

% Annealing schedule. 
% The algorithm run multiple runs of optimization with reducing smoothing parameter (scale) for each round. The annealing schedule helps the convergence of the optimization
% scale: starts at the value below and gradually decrease to zero, controls the size of the low-pass filter used to smooth the theta image. You can reduce the starting value of scale (to save time) if your data is of good quality (high sampling rate). Reducing it too much will result in convergence problems.
% anneal_step controls the rate of scale decreasing. 
scale = 2.4; % starting smoothing parameter
anneal_step = 0.4; % speed of scale decreasing

% convergence control in each round of optimization
% In each round, the EM iterations were run until convergence is achieved (controlled by cvge parameter) or maximum iteration had been reached.
cvge = 0.3; % convergence test criteria
max_iter = 5; %% maximum number of iteration for each round of optimization

%others
p = 0.2; % amplitude (sigms^2) of drift
eps = 0.001/h/w; % creep probability. Set to 0 if your system don't have a creep problem
max_shift = 30; % maximum drift (pixels) that is being calculated
resolution = 2; % Localization uncertainty (FWHM) in pixels

% ---------------------------------------------

% setting up parameters
if (nargin < 5)
    theta = construct_palm(O, h, w);
end

parameters.p = p;
parameters.eps = eps;
parameters.smooth = resolution;
parameters.max_shift = max_shift;
parameters.scale = scale;

%start
iter_r = 1;
d = zeros(length(O), 2);
while (scale >= 1.2)

    c = [0 0];
    iter = 0;
    display(['round - ' int2str(iter_r)]);

    while ((c(1) == 0 || c(2) == 0) && iter < max_iter)
        
        fs = round(exp(scale));

        S = BaSDI_iter(O, h, w, parameters, conv2(theta,ones(fs,fs))); 
        %parameters.smooth = resolution * exp(scale);
        %S = BaSDI_iter(O, h, w, parameters, theta); 

        theta = S.theta;
        iter = iter + 1;

        d_out = processing_result(S.g);
        c = testing_converge(d, d_out, cvge);
        d = d_out;
        %input('');
        
        %imagesc(theta); input('');

    end
    
    scale = scale - anneal_step;
    iter_r = iter_r + 1;

end

imagesc(S.theta);

