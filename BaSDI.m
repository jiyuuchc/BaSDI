function S = BaSDI(O, h, w)

% Change these to suit you need
max_iter = 5; % maximum number of iteration for each round of optimization
cvge = 0.3; % convergence test criteria

p = 0.2; % amplitude of drift
eps = 0; %0.001/h/w; % creep proability
max_shift = 30; % maximum drift that is being calculated
resolution = 2; % Localization uncertainty (FWHM) in PALM pixels

%annealing schedule
scale = 2.4; % starting smoothing parameter
anneal_step = 0.4; % amount of reduction of smoothness for each round

% ----------------------------------

% Preprocessing
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

