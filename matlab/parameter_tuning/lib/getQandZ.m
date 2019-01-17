function [Q1, Q2, Qc, Z, N, K] = getQandZ(imageRGBName, imageLabelsName, nrSP, beta)
% function computing Q1(constant binary terms), Q2(instensity dependent
% binary terms), and Qc(unary terms) for the data matrix Q = lambda1*Q1 +
% lambda2 * Q2 + Qc, as well as ground truth label matrix Z, number of
% nodes N and number of classes K.
% Date: 6/15/2018
% Authors: Siyi Hu

%% Superpixels
imageRGB = imread(imageRGBName);
imageLabels = imread(imageLabelsName);
imageGray = rgb2gray(imageRGB);

% get superpixels
[L, N] = superpixels(imageRGB, nrSP); 
spIndices = label2idx(L);

% Find connectivity between superpixels
connMatrix = findConnectivity(L, N);
fprintf('%i super pixels computed for image %s.\n', N, imageRGBName)

%% Get Q1 and Q2
% number of classes
K = double(max(max(imageLabels))) + 1; 
Q1 = zeros(N+K, N+K);
Q1(1:N, 1:N) = -connMatrix;

% Find the center and average intensity of each superpixel
spIntensity = getSPintensities(imageGray,spIndices,N);
Q2 = getQ2(connMatrix, N, K, beta, spIntensity);

%% Get Qc and Z
X_GT = getGroundTruth(spIndices, imageLabels, N, K);
% X = horzcat(ones(N,1), zeros(N, K-1));
X = getEstimatedLabels(spIndices, imageLabels, N, K);
fprintf('%i super pixels are not estimated correctly.\n', norm(X_GT-X, 'fro')^2/2)

Qc = zeros(N+K, N+K);
Qc(N+1:end, 1:N) = -0.5*X';
Qc(1:N, N+1:end) = -0.5*X;

Y = vertcat(X_GT, eye(K));
Z = Y*Y';
end
