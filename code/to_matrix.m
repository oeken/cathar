function R = to_matrix(instances,huge)
    u = 943;
    i = 1682;
    if ~huge
        u = max(instances(:,1));
        i = max(instances(:,2));
    end
    R = nan(u,i);
    N = length(instances);
    for j=1:N
        u_j = instances(j,1);
        i_j = instances(j,2);
        R(u_j,i_j) = instances(j,3);
    end
end