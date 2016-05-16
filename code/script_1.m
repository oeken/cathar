close all;
clc;
clear;
format bank;
rng(12345);
%%
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

%% READ TRAINING AND TEST DATA

filename = '../data/ml-100k/movs';
movie_names = importdata(filename);

% =====  Training Data  =====
filename = '../data/ml-100k/u1.base';
A = importdata(filename);
instances_trai = A(:,1:3);
R_trai = to_matrix(instances_trai,1);
% =====  Test Data  =====
filename = '../data/ml-100k/u1.test';
A = importdata(filename);
instances_test = A(:,1:3);
R_test = to_matrix(instances_test,1);

%% TRAIN & PLOT

acc = 0.001;    % convergence condition
err = 'rms';   % error calc method
l   = 3;       % latent factors
reg = 0.1;     % regulatization weight
bia = 1;

tic
[L_U, L_I, B_U, B_I, mu, iteration, e_all] = sgd(R_trai, ...
                                                 instances_trai, ...
                                                 l, ...
                                                 reg, ...
                                                 acc, ...
                                                 err, ...
                                                 bia);
toc 


%% LATENT & REGULARIZATION VS ERROR
% bia = 0
% bia_off_stats = zeros(10,7,2);
% for j=1:10
%     j
%     for k=1:7
%         k
%         tic
%         [L_U, L_I, B_U, B_I, mu, iteration, e_all] = sgd(R_trai, ...
%                                                            instances_trai, ...
%                                                            j, ...
%                                                            (k-1)/10, ...
%                                                            acc, ...
%                                                            err, ...
%                                                            bia);                
%         toc  
%         R_hat = mu + B_U + B_I + L_U * L_I';
%         bia_off_stats(j,k,1) = compute_error(R_trai, R_hat, err);
%         bia_off_stats(j,k,2) = compute_error(R_test, R_hat, err);
%     end
% end
% 
% bia = 1
% bia_on_stats = zeros(10,7,2);
% for j=1:10
%     j
%     for k=1:7
%         k
%         tic
%         [L_U, L_I, B_U, B_I, mu, iteration, e_all] = sgd(R_trai, 
%                                                            instances_trai, ...
%                                                            j, ...
%                                                            (k-1)/10, ...
%                                                            acc, ...
%                                                            err, ...
%                                                            bia);                
%         toc  
%         R_hat = mu + B_U + B_I + L_U * L_I';
%         bia_on_stats(j,k,1) = compute_error(R_trai, R_hat, err);
%         bia_on_stats(j,k,2) = compute_error(R_test, R_hat, err);
%     end
% end

%% COMPUTE ERROR

% R_hat1 = mu1 + B_U1 + B_I1 + L_U1 * L_I1';
% e_1 = compute_error(R_trai, R_hat1, err);
% e_2 = compute_error(R_test, R_hat1, err);

%% RECOMMENDATION
R_hat = mu + B_U + B_I + L_U * L_I';
u = 1;
liked = find(R_test(u,:) == 5);
disliked = find(R_test(u,:) == 1);
est_liked = R_hat(u,liked);
est_disliked = R_hat(u,disliked);

liked = find(R_test(u,:) == 5);
disliked = find(R_test(u,:) == 1);
unknown_liked_movies = movie_names(liked);
unknown_disliked_movies = movie_names(disliked);

liked = find(R_trai(u,:) == 5);
disliked = find(R_trai(u,:) == 1);
known_liked_movies = movie_names(liked);
known_disliked_movies = movie_names(disliked);

known = find(~isnan(R_trai(u,:)));
lk = length(known);
est = R_hat(u,:);
est(known) = nan;
[v, index] = sort(est,'descend');
recx = 50;
recommended = movie_names(index(lk+1:lk+recx+1));
unrecommended = movie_names(index(end-recx+1:end));

% opinion_on_recommended = R_test(index(lk+1:lk+recx+1));
% opinion_on_unrecommended = R_test(index(end-recx+1:end));

good_guess_on_like = intersect(recommended,unknown_liked_movies);
good_guess_on_dislike = intersect(unrecommended,unknown_disliked_movies);


%% PLOTS

ts = 16;
figure()
plot(1:length(e_all),e_all,'LineWidth',3);
title('Error (RMS) over Time (SGD)','FontSize',ts)
grid on
saveas(gcf, 'buff/1', 'png')

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

%% FIN
disp(' ')
disp('All done')