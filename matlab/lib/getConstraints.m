function [A, b] = getConstraints(N, K)
% function computing A and b for the linear equality constraint Ax=b
% x is binary
% Inputs:
% - N: number of superpixels
% - K: number of classes
% Date: 1/23/2018
% Authors: Luca Carlone, Siyi Hu

A = spalloc(N, N*K, N*K);
for i = 1:N
    A(i, (i-1)*K+1:i*K) = ones(1, K);
end
b = ones(N, 1);