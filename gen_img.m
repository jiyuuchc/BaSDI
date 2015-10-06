function I = gen_img(s)
% Generate the ground truth image of the size s x s

I = zeros(s,s);
I(round(s/2 - s/10):round(s/2 + s/10),round(s/2 - s/10):round(s/2 + s/10))=10;
I(round(s/2 - s/10):round(s/2 + s/10),round(s/2 - s/33): round(s/2 + s/33)) = 0;
I(round(s/2 - s/33): round(s/2 + s/33),round(s/2 - s/10):round(s/2 + s/10)) = 0;

