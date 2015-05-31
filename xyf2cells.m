function O = xyf2cells(xyf)
% O = xyf2cells(xyf)
% Convert an Nx3 array of x, y, frame# data to a cellarry
% The result is useful as an input for BaSDI

[h,w] = size(xyf);

%Do something useful if user mistakenly used a 3xN matrix instead of Nx3
if (w > 3 && h == 3) 
    h = w;
    w = 3;
    xyf = xyf';
end

%Well it's beyong hope
if (w < 3 )
    error('BaSDI:matrixSize', 'The input matrix size should be Nx3')
end

O = {};
frames = floor(xyf(:,3));
maxframe = max(frames);
minframe = min(frames);

cur = 1;
for f = minframe:maxframe
    idx = find(frames == f);
    O{cur} = xyf(idx, 1:2);
    cur = cur + 1;
end
