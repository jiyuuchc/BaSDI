function [d_out, sigma] = processing_result(g)

[h,w,f] = size(g);
maxshift = (h - 1)/2;
[x,y] = meshgrid(-maxshift:maxshift,-maxshift:maxshift);

for i = 1:f;
    gn = g(:,:,i);
    gn = gn / sum(gn(:));
    
    cx(i) = sum(sum(x.*gn));
    cy(i) = sum(sum(y.*gn));
    
    cx_2(i) = sum(sum((x.^2).*gn));
    cy_2(i) = sum(sum((y.^2).*gn));
    
    sdx = (cx_2(i)-(cx(i))^2);
    sdy = (cy_2(i)-(cy(i))^2);
    
    sigma(i) = sqrt(sdx + sdy);
end

%plot(-[cy - cy(1) ;cx - cx(1)]');
plot(-[cy ; cx]');
d_out = -[cy; cx]';
%plot(sigma);
