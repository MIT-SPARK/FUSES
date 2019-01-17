function BonnetAccuracy = getUnaryFactorAccuracy(hdfFiles, labelGTFiles);
% function computing accuracy of the unary factors
% Date: 7/11/2018
% Authors: Siyi Hu

BonnetAccuracy = zeros(1, length(hdfFiles));
for i = 1:length(hdfFiles)
    hdfFileName = hdfFiles{i};
    labelGTFileName = labelGTFiles{i};
    
    %% Read unary terms
    % get dimension of the problem
    data = h5read(hdfFileName,'/gm/numbers-of-states');
    N = length(data);
    K = data(1);
    K = cast(K, 'double');

    % get data matrices
    % unary terms
    data = h5read(hdfFileName,'/gm/function-id-16000/values');
    nrUnary = length(data)/K;
    indices = h5read(hdfFileName,'/gm/factors');
    g = zeros(N*K, 1);
    for j = 1:nrUnary
        node = indices(j*4) + 1; % index starts from 0 in hdf5
        g((node-1)*K+1 : node*K) = data((j-1)*K+1 : j*K);
    end
    G = reshape(g, K, N)';
    unaryLabel = -G*(0:K-1)';
    
    %% Compare with ground truth labels
    labelGT = csvread(labelGTFileName);
    BonnetAccuracy(1, i) = (1-nnz(labelGT - unaryLabel)/length(labelGT))*100;
end

