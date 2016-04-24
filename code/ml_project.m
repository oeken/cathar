close all;
clc;
clear;
format bank;
%rng(12344)

% generate synthetic data
known = 0.5;
i = 6;
u = 4;
l = 2;
R_full = round(rand([u,i]) * 5);
del  = randperm(u*i,round(u*i*(1-known)));
R = R_full;
R(del) = NaN;

% R = [5, 3, NaN, 1;
%      4, NaN, NaN, 1;
%      1, 1, NaN, 5;
%      1, NaN, NaN, 4;
%      NaN, 1, 5, 4];
% [u,i] = size(R);


% =====  Read Training Data  =====
filename = '../data/ml-100k/u1.base';
A = importdata(filename);
[m,n] = size(A);
instances = A(:,1:3);
% =====  Read Training Data  =====
u = 943;
i = 1682;
% u = instances(end,1);  % # of users
% i = instances(end,2);  % # of items
mu_all = mean(instances(:,3));
mu_u = nan(u,1);
mu_i = nan(1,i);
for k=1:u
   rows = find(instances(:,1) == k);
   mu_u(k) = mean(instances(rows,3));
end
for k=1:i
   rows = find(instances(:,2) == k);
   mu_i(k) = mean(instances(rows,3));
end

% mu_all = mean(nanmean(R));
% mu_u = nanmean(R,2);
% mu_i = nanmean(R,1);
bias_u = mu_u - mu_all;
bias_i = mu_i - mu_all;

% instances = find(~isnan(R) == 1);
U = rand(u,l);
I = rand(i,l);

eta = 0.9;
lambda = 0.05;
lambda2 = 0.05;
prev_error = Inf;
iteration = 1;
while true
   instance_count = length(instances);
   identity = speye(instance_count);
   P = identity(randperm(instance_count),:);  % permutation matrix
   shuffled_instances = P * instances;  % shuffles instances
   for j=1:instance_count
       current_instance = shuffled_instances(j);
       current_item = ceil(current_instance/u);
       current_user = mod(current_instance,u);
       if current_user == 0
           current_user = u;
       end
       current_l_user = U(current_user,:);
       current_l_item = I(current_item,:);
       
       r_j = R(current_user, current_item);
       y_j = mu_i(current_item) + bias_u(current_user) + current_l_user * current_l_item';
       e_j = r_j - y_j;
      
       update_u = eta * (e_j * current_l_item - lambda * current_l_user - lambda2 * bias_u(current_user));
       update_i = eta * (e_j * current_l_user - lambda * current_l_item - lambda2 * bias_i(current_item));
       U(current_user,:) = current_l_user + update_u;
       I(current_item,:) = current_l_item + update_i;
   end
   
   R_hat = repmat(mu_i,u,1) + repmat(bias_u,1,i) + U * I';
   Error = R - R_hat;
   error = sum(nansum(Error.^2));
   if abs(error - prev_error) < 0.0001
       break;
   end
   prev_error = error;
   iteration = iteration + 1;
end


R
R_hat = repmat(mu_i,u,1) + repmat(bias_u,1,i) + U * I'
Error = R - R_hat
iteration





