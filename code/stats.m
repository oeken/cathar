function [mu_all, mu_u, mu_i, bias_u, bias_i] = stats(instances)
    u = max(instances(:,1));
    i = max(instances(:,2));
    
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

    bias_u = mu_u - mu_all;
    bias_i = mu_i - mu_all;

end