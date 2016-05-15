true_R = magic(5);
hat_R = true_R;
tobe_nan = randperm(length(hat_R(:)),10);
hat_R(tobe_nan) = nan;