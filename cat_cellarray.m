function O1 = cat_cellarray(O, n)
% function O1 = cat_cellarray(O, n)
% combining every n cells together

for i = 1 : floor( (length(O)-1) / n) + 1
    O1{i} = cat(1, O{ (i-1)*n + 1 : (i-1)*n + n });
end

