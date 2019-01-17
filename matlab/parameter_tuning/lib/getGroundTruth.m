function X = getGroundTruth(spIndices, imageLabels, N, K)
% function generating ground-truth labels as a matrix of size N x K 
% Date: 6/15/2018
% Authors: Siyi Hu

X = zeros(N, K);

for i = 1:N
    pixels = spIndices{i}; 
    k = mode(imageLabels(pixels)) + 1; % the dominant class k in this SP
    X(i, k) = 1;
end
end