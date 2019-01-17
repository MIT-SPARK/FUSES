% Script fitting parameters lambda1 and lambda2 using KKT conditions
% Objective function: 
%   sum of (primal - dual)^2 + sum of ||complementary slackness||_F
% Date: 1/24/2018
% Authors: Luca Carlone, Siyi Hu

clear all; close all; % clc;
addpath('./lib') 

%% Inputs
% parameters
testDataset = 'aachen'; % dataset name 
T = 1;                 % number of images
nrSP = 200;            % number of super pixels
beta = 2.5*10^-4;
figureOn = false;

%% Compute segmentation
Q1List = cell(1, T);
Q2List = cell(1, T);
QcList = cell(1, T);
ZList = cell(1, T);
cList = cell(1, T);
FList = cell(1, T);

for j = 1:T
    [imageRGBName, imageLabelsName] = CityscapesParser(testDataset, j);
    [Q1, Q2, Qc, Z, N, K] = getQandZ(imageRGBName, imageLabelsName, nrSP, beta);
    [c, F] = getCandF(N, K);
    
    Q1List{j} = Q1;
    Q2List{j} = Q2;
    QcList{j} = Qc;
    ZList{j} = Z;
    cList{j} = c;
    FList{j} = F;
end

%% Formulate minimization problem
cvx_quiet(true)
cvx_solver mosek
cvx_begin 
variables lambda1(1) lambda2(1) x(N+10+K^2, T)

obj = 0;
for j = 1:T
    N = size(Q1List{j}, 1) - K; % update N for frame j

    sum = 0;
    for i=1:N+K^2
        sum = sum + x(i, j)*FList{j}{i};
    end
    obj = obj + ( cList{j}'*x(1:length(cList{j}), j) + ...
        lambda1*trace(Q1List{j}*ZList{j}) + ...
        lambda2*trace(Q2List{j}*ZList{j}) + ...
        trace(QcList{j}*ZList{j}) )^2 + ...
        norm((lambda1*Q1List{j} + lambda2*Q2List{j} + QcList{j} + sum)*ZList{j}, 2);
end

minimize( obj )
subject to
    for j = 1:T
        N = size(Q1List{j}, 1) - K; % update N for frame j
        
        sum = 0;
        for i=1:N+K^2
            sum = sum + x(i, j)*FList{j}{i};
        end
        lambda1*Q1List{j} + lambda2*Q2List{j} + QcList{j} + sum == semidefinite(N+K);
    end

timeCVX = tic;
cvx_end
time_solver = toc(timeCVX);

%% Display results
disp(['lambda1 = ' num2str(lambda1)])
disp(['lambda2 = ' num2str(lambda2)])
disp(['Time elapsed: ' num2str(time_solver) ' s'])

%% Compute segmentation
for j = 1:T
    Q = lambda1*Q1List{j}+lambda2*Q2List{j}+QcList{j};
    N = size(Q, 1) - K;
    dataMatrices.Hcompact = Q(1:N, 1:N);
    dataMatrices.Gcompact = Q(1:N, N+1:N+K)*2;
    
    [imageRGBName, imageLabelsName] = CityscapesParser(testDataset, j);
    [cplexOut, sdp4Out, manoptOut] = ...
        computeMultiClassSeg(imageRGBName, imageLabelsName, nrSP, dataMatrices, figureOn);
    
    x_GT = reshape(ZList{j}(1:N, N+1:end)', [], 1);
    x_unary = reshape(QcList{1}(1:N, N+1:end)'*-2, [], 1);
    fprintf("Gap between cplex and manopt: %i\n", norm(cplexOut.x-manoptOut.x, 1)/2);
    fprintf("Gap between ground-truth and manopt: %i\n", norm(x_GT-manoptOut.x, 1)/2);
    fprintf("Gap between prediction and manopt: %i\n", norm(x_unary-manoptOut.x, 1)/2);
end
