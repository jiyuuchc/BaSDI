function I = gen_img(s)
I = zeros(s,s);
I(round(s/2 - s/10):round(s/2 + s/10),round(s/2 - s/10):round(s/2 + s/10))=10;
I(round(s/2 - s/10):round(s/2 + s/10),round(s/2 - s/33): round(s/2 + s/33)) = 0;
I(round(s/2 - s/33): round(s/2 + s/33),round(s/2 - s/10):round(s/2 + s/10)) = 0;

