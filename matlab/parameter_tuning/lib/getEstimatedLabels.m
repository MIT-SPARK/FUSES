function X = getEstimatedLabels(spIndices, imageLabels, N, K)
% function generating estimated labels as a matrix of size N x K 
% (The estimated label comes from picking a pixel randomly from the
% superpixel and assigning ground-truth label of that pixel to the
% superpixel)
% Date: 6/15/2018
% Authors: Siyi Hu

X = zeros(N, K);

rng(0)
for i = 1:N
    pixels = spIndices{i}; 
    index = randi(length(pixels));
    k = imageLabels(pixels(index)) + 1;
    X(i, k) = 1;
end
end