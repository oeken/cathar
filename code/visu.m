close all;
clc;
clear;
format bank;
rng(12345);







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
% instances_trai = to_instances(R_trai);
% instances_test = to_instances(R_test);


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





acc = 0.001; % convergence condition
l = 4;       % latent factors
r = 0.0;     % regulatization weight

tic
[L_U1, L_I1, B_U1, B_I1, mu1, iteration1, e_all1] = sgd_new(R_trai, instances_trai, l, r, acc);
toc 

R_hat1 = mu1 + B_U1 + B_I1 + L_U1 * L_I1';
e_1 = compute_error(R_trai, R_hat1, 'rms');
e_2 = compute_error(R_test, R_hat1, 'rms');


ts = 16;

figure()
plot(1:length(e_all1),e_all1,'LineWidth',3);
title('Error (RMS) over Time (SGD)','FontSize',ts)
grid on
saveas(gcf, 'buff/1', 'png')



liked = find(R_test(1,:) >= 4);
disliked = find(R_test(1,:) <= 2);
est_liked = R_hat1(1,liked)
est_disliked = R_hat1(1,disliked)


%%% 1 %%%
% figure()
% % plot(L_U1(:,1),L_U1(:,2),'x','LineWidth',3);
% scatter3(L_U1(:,1),L_U1(:,2),L_U1(:,3),'x','LineWidth',3)
% title('Latent Factor Scatter Plot for Users (SGD)','FontSize',ts)
% grid on
% saveas(gcf, 'buff/users', 'png')

%%% 2 %%%
% figure()
% % plot(L_I1(:,1),L_I1(:,2),'x','LineWidth',3);
% scatter3(L_I1(:,1),L_I1(:,2),L_I1(:,3),'x','LineWidth',3)
% title('Latent Factor Scatter Plot for Movies (SGD)','FontSize',ts)
% grid on
% saveas(gcf, 'buff/movs', 'png')




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



