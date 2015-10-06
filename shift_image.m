function J = shift_image(I, d)
% J = shift_image(I, d)
% Shift image I by a translational drift d
% I: input image. 2D array
% d: tuple (dx, dy)
% J: output image.

[h,w] = size(I);
dx = d(2);
dy = d(1);

if (dx < 0)
    x1_s = - dx + 1; x2_s = w;
    x1_d = 1; x2_d = w + dx;
else
    x1_s = 1; x2_s = w - dx;
    x1_d = dx + 1; x2_d = w;
end

if (dy < 0)
    y1_s = - dy + 1; y2_s = h;
    y1_d = 1; y2_d = h + dy;
else
    y1_s = 1; y2_s = h - dy;
    y1_d = dy + 1; y2_d = h;
end

J = zeros(h,w);
J(y1_d:y2_d, x1_d:x2_d) = I(y1_s:y2_s, x1_s:x2_s);
