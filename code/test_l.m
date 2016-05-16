close all;
clc;
clear;
format bank;
rng(12345);

l = 2;      % latent factors
r = 0;      % regulatization on/off
k = 0.9;    % known percentage
b = 0;      % biases on/off
e = 'mse';  % error function


%read the excel file
% filename='toy_movies.xlsx';
% [num,txt,raw]=xlsread(filename);
% R_true = num';
% R_trai = R_true;
% R_test = R_true;
% [u,i] = size(R_true);
% trai_size = 0.9;
% test = randperm(u*i,u*i*trai_size);
% trai = setdiff(1:u*i,test);
% R_trai(trai) = nan;
% R_test(test) = nan;


% =====  Read Training Data  =====
filename = '../data/ml-100k/u1.base';
A = importdata(filename);
instances_trai = A(:,1:3);
R_trai = to_matrix(instances_trai,1);
% =====  Read Training Data  =====


% =====  Read Test Data  =====
filename = '../data/ml-100k/u1.test';
A = importdata(filename);
instances_test = A(:,1:3);
R_test = to_matrix(instances_test,1);
% =====  Read Test Data  =====


acc = 0.01;  % convergence condition
ts = 16;

max_l = 10;
e_sgd_trai = zeros(max_l,1);
e_als_trai = zeros(max_l,1);

e_sgd_test = zeros(max_l,1);
e_als_test = zeros(max_l,1);

for j=1:max_l
    l = j
    tic
    [L_U1, L_I1, iteration1, e_all1] = sgd(R_trai, instances_trai, l, acc);
    toc 
%     tic
%     [L_U2, L_I2, e_all2] = als_manual(R_trai, instances_trai, l, 2);
%     toc
    R_hat1 = L_U1 * L_I1';
%     R_hat2 = L_U2 * L_I2';
    
    e_sgd_trai(j) = compute_error(R_trai, R_hat1, e);
%     e_als_trai(j) = compute_error(R_trai, R_hat2, e);
    
    e_sgd_test(j) = compute_error(R_test, R_hat1, e);
%     e_als_test(j) = compute_error(R_test, R_hat2, e);

end





%%% 1 %%%
figure()
% len_e1 = length(e_all1);
plot(1:max_l,e_sgd_trai,'LineWidth',3);
hold on;
plot(1:max_l,e_sgd_test,'LineWidth',3);
title('Error (MSE) vs # of Latent Factors','FontSize',ts)
grid on
saveas(gcf, 'buff/l_sgd', 'png')

% %%% 2 %%%
% figure()
% % len_e1 = length(e_all1);
% plot(1:max_l,e_als_trai,'LineWidth',3);
% hold on;
% plot(1:max_l,e_als_test,'LineWidth',3);
% title('Error (MSE) vs # of Latent Factors','FontSize',ts)
% grid on
% saveas(gcf, 'buff/l_als', 'png')

%%% 2 %%%
% figure()
% len_e2 = length(e_all2);
% plot(1:len_e2,e_all2,'LineWidth',3);
% title('Error (MSE) over Time (ALS)','FontSize',ts)
% saveas(gcf, 'buff/6', 'png')

disp(' ')
disp('All done')



