% Script fitting parameters lambda1 and lambda2 using KKT conditions
% Objective function: sum of ||Z_GT - Z_estimated||_F
% Note: this formulation is NOT CONVEX
% Date: 1/24/2018
% Authors: Luca Carlone, Siyi Hu

clear all; close all; % clc;
addpath('./lib') 

%% Inputs
% parameters
testDataset = 'train'; % dataset name 
T = 1;                 % number of images
nrSP = 100;            % number of super pixels
beta = 2.5*10^-4;
figureOn = true;

%% Compute segmentation
Q1List = cell(1, T);
Q2List = cell(1, T);
QcList = cell(1, T);
ZList = cell(1, T);
cList = cell(1, T);
FList = cell(1, T);

for j = 1:T
    [imageRGBName, imageLabelsName] = DAVISParser(testDataset, j);
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
variables lambda1(1) lambda2(1) x(N+K^2, T) 
variable Z(N+K, N+K) symmetric

obj = 0;
for j = 1:T
    obj = obj + norm(ZList{j} - Z, 'fro');
end

minimize( obj )
subject to
    Z = semidefinite(N+K);
    for j = 1:T
        N = size(Q1List{j}, 1) - 5; % update N for frame j
        
        sum = 0;
        for i=1:N+K^2
            sum = sum + x(i, j)*FList{j}{i};
        end
        lambda1*Q1List{j}+lambda2*Q2List{j}+QcList{j} + sum == semidefinite(N+K);
        (lambda1*Q1List{j}+lambda2*Q2List{j}+QcList{j} + sum)*Z == 0;
        cList{j}'*x(1:length(cList{j}), j) + ...
            trace((lambda1*Q1List{j}+lambda2*Q2List{j}+QcList{j})*Z) == 0;
    end

timeCVX = tic;
cvx_end
time_solver = toc(timeCVX);

%% Display results
disp(['lambda1 = ' num2str(lambda1)])
disp(['lambda2 = ' num2str(lambda2)])
disp(['Time elapsed: ' num2str(time_solver) ' s'])
pause

%% Compute segmentation
for j = 1:T
    Q = lambda1*Q1List{j}+lambda2*Q2List{j}+QcList{j};
    dataMatrices.Hcompact = Q(1:N, 1:N);
    dataMatrices.Gcompact = Q(1:N, N+1:N+K)*2;
    
    [cplexOut, sdp4Out, manoptOut] = ...
        computeMultiClassSeg(testDataset, j, nrSP, dataMatrices, figureOn);
end
