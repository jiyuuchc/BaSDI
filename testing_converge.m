function c = testing_converge(t, t1, eps)
% c = testing_converge(t, t1, eps)
% testing if the relative distance between two vector t and t1 is smaller than eps

if (nargin < 3)
    eps = 0.001;
end

dt = std(t - t1);
c = ( dt < std(t1) * eps );
