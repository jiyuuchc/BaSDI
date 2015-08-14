function S = BaSDI(O, pixel_size)
% Bayesian Super-resolution Drift Inference
% S = BaSDI(O, pixel_size)
% O: input localization data
%    'O' can either be a simple array or a cell array
%    If O is a simple array, each row of the array represents one
%    localization event in the format of (y,x,frame). Frame starts at 1.
%    If O is a cell array, each cell is a two-colume array representing a
%    single image frame.
%    The y,x coordiantes can be in any physical unit, e.g. 'nm', as long as
%    it is the same unit for 'pixel_size'. Coordinate (0,0) represents top-left 
%    of the image.
% pixel_size: The pixle_size used for rendering the final corrected
% super-resolution image. 
%
% S: output structure. 
%    S.theta: the corrected image.
%    S.g:	  posterior distirbution funciton P(d_k|o,theta).
%    S.e:     Drift probability of each frame w/o considering prior
%    probability distribution P(d). Can be used as the input for
%    compute the most likely drift trace using viterbi.m.

if (nargin == 1) 
    pixel_size = 1;
    warning('BaSDI:Preprocessing', 'Only one input. Assuming pixel_size is 1');
end

% convert the array into an cell array
if ~iscell(O)
    if ~ismatrix(O)
        error('BaSDI:Preprocessing', 'Wrong input format');
    end
    
    [oh,ow] = size(O);
    if (ow ~= 3)
        error('BaSDI:Preprocessing', 'Wrong input format');        
    end
    
    max_frame = max(O(:,end));
    OC = {};
    for i = 1:max_frame
        OC{i} = O( find(floor(O(:,end))==i), 1:2 );
    end
    
    O = OC;
end

% Estimating image size and convert coordinates into pixels

padding = 30; % padding some empty pixels around image borders

L = length(O);
mc = [0,0];
for i = 1:L;
    O{i} = floor( O{i} / pixel_size ) + padding;
    mc = max(mc, max(O{i},[],1));
end

if (mc(1) < 100 || mc(2) < 100)

end

% round up to the near 10.
mc = (floor(mc/10) + 1) * 10;
h = mc(1) + padding
w = mc(2) + padding

if (h*w > 5e7) 
    warning('BaSDI:Preprocessing', 'Very large images. May take very long time');
end

S = BaSDI_main(O, h, w);

