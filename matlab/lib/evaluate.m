function fval = evaluate(filename, label)
% function evaluating the cost of a CRF stored in an hdf5 file for a given 
% label vector
% Date: 7/5/2018
% Authors: Siyi Hu

%% Read CRF data
% get dimension of the problem
data = h5read(filename,'/gm/numbers-of-states');
N = length(data);
K = data(1);
K = cast(K, 'double');

% check label dimension
if length(label) ~= N
    error('Label size does not match the number of unary terms in the CRF')
elseif max(label) >= K
    error('label class exceeding the number of classes in the CRF')
end

% get data matrices
% unary terms
data = h5read(filename,'/gm/function-id-16000/values');
nrUnary = length(data)/K;
indices = h5read(filename,'/gm/factors');
g = zeros(N*K, 1);
for i = 1:nrUnary
    node = indices(i*4) + 1; % index starts from 0 in hdf5
    g((node-1)*K+1 : node*K) = g((node-1)*K+1 : node*K) + data((i-1)*K+1 : i*K);
end
G = reshape(g, K, N)';

% binary terms
% first 4*nrUnary entries are for the unary terms
indices = indices(4*nrUnary+1:end); % indices for binary terms
data = h5read(filename,'/gm/function-id-16006/values');
% binary terms each takes 5 entries
nrBinary = length(indices)/5;
H = spalloc(N, N, nrBinary*2);
% H = zeros(N, N);
for i = 1:nrBinary
    n1 = indices((i-1)*5+4) + 1; % index starts from 0 in hdf5
    n2 = indices((i-1)*5+5) + 1;
    H(n1, n2) = data((i-1)*2+1);
end
H = (H+H')./2;

%% Read label
X = zeros(N, K);
for i = 1:N
    class = label(i) + 1; % label starts from 0
    X(i, class) = 1;
end

fval = trace(H*X*X') + trace(G*X');
end

