function [L_U, L_I, iteration, e_all] = sgd(R, l, acc)
%     instances = to_instances(R);
%     N = length(instances);
    N = 80000;
%     [mu_all, mu_u, mu_i, bias_u, bias_i] = stats(instances);

%     u = max(instances(:,1));
%     i = max(instances(:,2));
    u = 943;
    i = 1682;
    
    L_U = rand(u,l);
    L_I = rand(i,l);

    eta = 0.01;  % learning rate
    lambda = 0.0;
    lambda2 = 0.00;


    b_size = 5;
    e_buffer = inf(b_size,1);
    iteration = 1;
    e_all = [];
    
    while true
        I = speye(N);
        P = I(randperm(N),:);  % permutation matrix
        shuffled_instances = P * instances;  % shuffles instances
        for j=1:N
            current_instance = shuffled_instances(j,:);
            user_j = current_instance(1);
            item_j = current_instance(2);
            r_j = current_instance(3);
            user_j_l = L_U(user_j,:);
            item_j_l = L_I(item_j,:);
            %        y_j = mu_i(item_j) + bias_u(user_j) + user_j_l * item_j_l';
            y_j = user_j_l * item_j_l';
            e_j = r_j - y_j;

            update_u = eta * (e_j * item_j_l - lambda * user_j_l - lambda2 * bias_u(user_j));
            update_i = eta * (e_j * user_j_l - lambda * item_j_l - lambda2 * bias_i(item_j));
            L_U(user_j,:) = user_j_l + update_u;
            L_I(item_j,:) = item_j_l + update_i;
        end


        %    R_hat = repmat(mu_i,u,1) + repmat(bias_u,1,i) + L_U * L_I';

        e = compute_error(R, L_U*L_I','mse');
        e_all = [e_all e];
        e_buffer = circshift(e_buffer,-1);
        e_buffer(end) = e;

        converged = 0;
        if mod(iteration,b_size) == 0
            converged = has_converged(e_buffer, 2, acc);
        end
        if converged
            break;
        end
        iteration = iteration + 1;
    end

    %         if mod(iteration,5) == 0
    %             iteration
    %             error = nansum(E(:).^2)
    %             error / prev_error
    %         end
end