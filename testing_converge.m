function c = testing_converge(t, t1, eps)

if (nargin < 3)
    eps = 0.001;
end

dt = std(t - t1);
c = ( dt < std(t1) * eps );
