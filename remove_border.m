function OC = remove_border(O, h, w, m)

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
