I = gen_img(500) + 2;
d = (cumsum(randn(1000,2)))/4;
O = gen_palm_data(I, 50, d);

imagesc(construct_palm(O,500,500));
input('showing uncorrected image\n press enter to continue');

display('running BaSDI');
S = BaSDI_main(O, 500, 500);
