function [c, FList] = getCandF(N, K)
% function generating vector c and a list of Fi for KKT conditions 
% Date: 6/15/2018
% Authors: Siyi Hu

c = vertcat(ones(N+K, 1), zeros(K^2-K, 1));

FList = cell(1, N+K^2);
for i = 1:N+K
    Fi = zeros(N+K, N+K);
    Fi(i, i) = 1;
    FList{i} = Fi;
end

count = N+K;
for i = 1:K
    for j = 1:K
        if i~=j
            count = count + 1;
            Fi = zeros(N+K, N+K);
            Fi(N+i, N+j) = 1;
            FList{count} = Fi;
        end
    end
end