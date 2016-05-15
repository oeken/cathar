function [true_r, hat_r] = generate_data(u, i, p)
    true_r = randi(5,u,i);
    N = u * i ;
    k = ceil(N*(1-p));
    hat_r = true_r;
    hat_r(randperm(N,k)) = nan; 
end