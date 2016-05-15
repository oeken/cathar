function e = compute_error(R, R_hat, method)        
    E = R - R_hat;
    E_vec = E(:);
    e = nanmean(E_vec.^2);                       
    if strcmp(method,'rms')
        e = sqrt(e);           
    end    
end