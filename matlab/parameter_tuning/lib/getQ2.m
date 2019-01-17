function Q2 = getQ2(connMatrix, N, K, beta, spIntensity)
% function computing Q2 such that Q2(i, j) = exp(-beta*(I_i-I_j)^2)
% for each neighboring superpixel pair
% Date: 6/15/2018
% Authors: Siyi Hu

Q2 = spalloc(N+K, N+K, sum(connMatrix(:))*2);
for i = 1:N
    for j = 1:i-1 % connMatrix is symmetric so we only need lower triangular part (no diagonal) 
        if connMatrix(i,j) == 1
            % average intensity of superpixel i and j
            Ii = spIntensity(i);
            Ij = spIntensity(j);
            Q2(i,j) = -exp(-(Ii-Ij)^2*beta);
        end 
    end
end
Q2 = Q2 + Q2'; % H is symmetric
end