function instances = to_instances(R)    
    [u,i] = find(~isnan(R));
    s = length(u);
    instances = [u i zeros(s,1)];
    for j=1:s
        instances(j,3) = R(u(j),i(j));
    end
end