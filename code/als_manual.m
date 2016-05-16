function [L_U, L_I, e_all] = als_manual(R, instances, l, times)
    
    N = length(instances);
    u = max(instances(:,1));
    i = max(instances(:,2));

    L_U = rand(u,l);
    L_I = rand(i,l);

    A = L_U;
    B = L_I';    
    e = compute_error(R, A*B,'mse');
    e_all = e;        
    
    for j=1:times
        b_mat = zeros(N,u*l);
        for k=1:N
            temp = instances(k,1:2);
            x = temp(1);
            y = temp(2);
            locations = 1+(x-1)*l : x*l;
            values = B(:,y);
            b_mat(k,locations) = values;
        end
        A = reshape(b_mat\instances(:,3),[l,u]);
        A = A';

        
        a_mat = zeros(N,i*l);
        for k=1:N
            temp = instances(k,1:2);
            x = temp(1);
            y = temp(2);
            locations = 1+(y-1)*l : y*l;
            values = A(x,:);
            a_mat(k,locations) = values;
        end
        B = reshape(a_mat\instances(:,3),[l,i]);
       
        
        e = compute_error(R, A*B,'mse');
        e_all = [e_all e];               
    end
    L_U = A;
    L_I = B';
end