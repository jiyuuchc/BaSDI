function OC = remove_border(O, h, w, m)
% OC = remove_border(O, h, w, m)
% Remove localization data that are within a m pixel border of the images.
% O: localization dataset. Cell array of N elements.
% h: height of image.
% w: width of image.
% m: border size.
% OC: new localization dataset with molecules at the border removed.

for i = 1:length(O)
    o = O{i};
    len = size(o,1);
    oc = [];
    for j = 1:len;
        x = o(j,1);
        y = o(j,2);
        if (x>=m && x < w-m && y >= m && y < h - m)
            oc = [oc; x,y];
        end
    end
    OC{i}= oc;
end
