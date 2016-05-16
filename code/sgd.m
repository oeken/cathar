function [L_U, L_I, B_U, B_I, mu, iteration, e_all] = sgd(R, ...
                                                          instances, ...
                                                          l, ...
                                                          reg, ...
                                                          acc, ...
                                                          err, ...
                                                          bia)
    
    N = length(instances);       
    u = max(instances(:,1));
    i = max(instances(:,2));        
    
    L_U = rand(u,l);
    L_I = rand(i,l);
    if bia
        B_U = rand(u,1);
        B_I = rand(i,1);
        mu = mean(instances(:,3));  % average of all ratings
    else
        B_U = zeros(u,1);
        B_I = zeros(i,1);
        mu = 0;
    end
        
    % loss fn = (r - b_u - b_i - mu - qp)^2
    %           + lambda * (q^2 + p^2 + b_u^2 + b_i^2)
    
    eta = 0.01;  % learning rate    
    lambda = reg;  % weight of regularization

    b_size = 5;
    iteration = 1;
    R_hat = mu + repmat(B_U,1,i) + repmat(B_I,1,u)' + L_U * L_I';
    e = compute_error(R, R_hat, err);
    e_all = e;
    
    while true
        I = speye(N);
        P = I(randperm(N),:);  % permutation matrix
        shuffled_instances = P * instances;  % shuffle instances
        for j=1:N
            current_instance = shuffled_instances(j,:);
            user = current_instance(1);
            item = current_instance(2);
            r = current_instance(3);  % desired output
            
            user_fact = L_U(user,:);
            item_fact = L_I(item,:);
            user_bias = B_U(user);
            item_bias = B_I(item);
            
            y = mu + user_bias + item_bias + user_fact * item_fact';  % predicted output
            e = r - y;  % error

            update_user_fact = eta * (e * item_fact - lambda * user_fact);
            update_item_fact = eta * (e * user_fact - lambda * item_fact);
            
            if bia
                update_user_bias = eta * (e - lambda * user_bias);
                update_item_bias = eta * (e - lambda * item_bias);
            end
                                                    
            L_U(user,:) = user_fact + update_user_fact;
            L_I(item,:) = item_fact + update_item_fact;            
            
            if bia
                B_U(user) = user_bias + update_user_bias;
                B_I(item) = item_bias + update_item_bias;
            end
            
        end
       
        R_hat = mu + repmat(B_U,1,i) + repmat(B_I,1,u)' + L_U * L_I';
        e = compute_error(R, R_hat, err);
        e_all = [e_all e];     

        converged = 0;
        if mod(iteration,b_size) == 0            
            e_buffer = e_all(end-b_size+1:end)';  % last <b_size> elements
            converged = has_converged(e_buffer, 2, acc);
        end
        if converged
            break;
        end
        iteration = iteration + 1;
    end
    B_U = repmat(B_U,1,i);
    B_I = repmat(B_I,1,u)';
end