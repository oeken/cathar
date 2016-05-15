close all;
clc;
clear;
format bank;
rng(12345);


% =====  Read Training Data  =====
% filename = '../data/ml-100k/u1.base';
% A = importdata(filename);
% instances = A(:,1:3);
% % instances = instances(1:2000,:);
% R = to_matrix(instances,0);
% =====  Read Training Data  =====

% [R_true, R] = generate_data(6,6,0.5);

%read the excel file
filename='toy_movies.xlsx';
[num,txt,raw]=xlsread(filename);
R_true = num';
R = R_true;
[u,i] = size(R_true);
% to_delete = randi(u*i,u*i*0.5,1);
% R(to_delete) = nan;

users = txt(2,3:22);
movies = txt(3:22,2);
l = 1;  % latent factors
acc = 0.001;  % accuracy


tic
disp('SGD executes')
[L_U1, L_I1, iteration1, e_all1] = sgd(R, l, acc);
disp('SGD done')
toc


tic
disp(' ')
disp('ALS executes')
[L_U2, L_I2, iteration2, e_all2] = als(R, l, acc);
disp('ALS done')
toc


tic
disp(' ')
disp('SVD executes')
[U,S,V] = svd(R);
L_U3 = U(:,1:l);
L_I3 = V(:,1:l);
s = S(1:l,1:l);
disp('SVD done')
toc

R_hat1 = L_U1 * L_I1';
R_hat2 = L_U2 * L_I2';
R_hat3 = L_U3 * s * L_I3';
compute_error(R, R_hat1, 'mse')
compute_error(R, R_hat2, 'mse')
compute_error(R, R_hat3, 'mse')


fs = 8;
ts = 16;
sc = 0.025;
m = 1;
dummy = zeros(i,1);
dummy(:,:) = 0.6+(sc*m);
%%% 1 %%%
figure()
% plot(L_I1(:,1),L_I1(:,2),'x','LineWidth',10)
plot(L_I1(:,1),0.1,'x','LineWidth',10)
% m = max(L_I1(:,2)) - min(L_I1(:,2));
% text(L_I1(:,1),L_I1(:,2)+(sc * m), movies, ...
text(L_I1(:,1),dummy, movies, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',fs, ...
            'FontWeight','normal','Rotation',90);
grid on
title('Latent Factor Scatter Plot For Movies (SGD)','FontSize',ts)
saveas(gcf, 'buff/1', 'png')


%%% 2 %%%
figure()
% plot(L_I2(:,1),L_I2(:,2),'x','LineWidth',10)
plot(L_I2(:,1),0.1,'x','LineWidth',10)
% m = max(L_I2(:,2)) - min(L_I2(:,2));
% text(L_I2(:,1),L_I2(:,2)+(sc * m), movies, ...
text(L_I2(:,1),dummy, movies, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',fs, ...
            'FontWeight','normal','Rotation',90);
grid on
title('Latent Factor Scatter Plot For Movies (ALS)','FontSize',ts)
saveas(gcf, 'buff/2', 'png')

%%% 3 %%%
figure()
% plot(L_I3(:,1),L_I3(:,2),'x','LineWidth',10)
plot(L_I3(:,1),0.1,'x','LineWidth',10)
% m = max(L_I3(:,2)) - min(L_I3(:,2));
% text(L_I3(:,1),L_I3(:,2)+(sc * m), movies, ...
text(L_I3(:,1),dummy, movies, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',fs, ...
            'FontWeight','normal','Rotation',90);
grid on
title('Latent Factor Scatter Plot For Movies (SVD)','FontSize',ts)
saveas(gcf, 'buff/svd_mov', 'png')

        

%%% 4 %%%
figure()
% plot(L_U1(:,1),L_U1(:,2),'x','LineWidth',10)
plot(L_U1(:,1),0.1,'x','LineWidth',10)
% m = max(L_U1(:,2)) - min(L_U1(:,2));
% text(L_U1(:,1),L_U1(:,2)+(sc * m), users, ...
text(L_U1(:,1),dummy, users, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',fs, ...
            'FontWeight','normal','Rotation',90);
grid on
title('Latent Factor Scatter Plot For Users (SGD)','FontSize',ts)
saveas(gcf, 'buff/3', 'png')


%%% 5 %%%
figure()
% plot(L_U2(:,1),L_U2(:,2),'x','LineWidth',10)
plot(L_U2(:,1),0.1,'x','LineWidth',10)
% m = max(L_U2(:,2)) - min(L_U2(:,2));
% text(L_U2(:,1),L_U2(:,2)+(sc * m), users, ...
text(L_U2(:,1),dummy, users, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',fs, ...
            'FontWeight','normal','Rotation',90);
grid on
title('Latent Factor Scatter Plot For Users (ALS)','FontSize',ts)
saveas(gcf, 'buff/4', 'png')

%%% 6 %%%
figure()
% plot(L_U3(:,1),L_U3(:,2),'x','LineWidth',10)
plot(L_U3(:,1),0.1,'x','LineWidth',10)
% m = max(L_U3(:,2)) - min(L_U3(:,2));
% text(L_U3(:,1),L_U3(:,2)+(sc * m), users, ...
text(L_U3(:,1),dummy, users, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',fs, ...
            'FontWeight','normal');
grid on
title('Latent Factor Scatter Plot For Users (SVD)','FontSize',ts)
saveas(gcf, 'buff/svd_user', 'png')



%%% 7 %%%
figure()
len_e1 = length(e_all1);
plot(1:len_e1,e_all1,'LineWidth',3);
title('Error (MSE) over Time (SGD)','FontSize',ts)
saveas(gcf, 'buff/5', 'png')


%%% 8 %%%
figure()
len_e2 = length(e_all2);
plot(1:len_e2,e_all2,'LineWidth',3);
title('Error (MSE) over Time (ALS)','FontSize',ts)
saveas(gcf, 'buff/6', 'png')


disp(' ')
disp('All done')

% text(L_I1(:,1),L_I(:,2),zeros(i,1)+0.05,movies,'Rotation',90)
% ylim([-0.1,1.2])

% figure()
% plot(L_I2(:,1),0,'x')
% text(L_I2(:,1),zeros(i,1)+0.05,movies,'Rotation',90)
% ylim([-0.1,1.2])




