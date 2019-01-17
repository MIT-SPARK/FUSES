function Hcompact = binaryTermsCompact(N, connMatrix, parameters, spIntensity)
% function computing matrix H in the energy function
% Inputs:
% - N: number of superpixels
% - connMatrix: matrix storing nearby pixel pairs
% - R, C: row and column of superpixel centroids
% - parameters: [lambda1 lambda2 beta] values for computing cost
% - spIntensity: average intensity of superpixels
% Date: 1/23/2018
% Authors: Luca Carlone, Siyi Hu

% extract parameters
lambda1 = parameters(1);
lambda2 = parameters(2);
beta = parameters(3);

% formulate H matrix
Hcompact = spalloc(N, N, sum(connMatrix(:))*2);
for i = 1:N
    for j = 1:i-1 % connMatrix is symmetric so we only need lower triangular part (no diagonal) 
        if connMatrix(i,j) == 1
            % average intensity of superpixel i and j
            Ii = spIntensity(i);
            Ij = spIntensity(j);
            
            % the cost of i and j having different labels
            f = lambda1 + lambda2*exp(-(Ii-Ij)^2*beta);
            
            % add cost to H
            % if vectors xi and xj are the same, energy is -f
            % otherwise, energy is 0
            % this creates an energy difference of f
            Hcompact(i,j) = -f;
        end 
    end
end
Hcompact = Hcompact + Hcompact'; % H is symmetric
end