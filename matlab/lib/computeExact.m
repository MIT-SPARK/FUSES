function [label, fval_cplex, offset] = computeExact(filename)
% function computing the exact MRF solution using cplex given a hdf5 file
% Date: 6/25/2018
% Authors: Siyi Hu

%% Read data
% get dimension of the problem
data = h5read(filename,'/gm/numbers-of-states');
N = length(data);
K = data(1);
K = cast(K, 'double');

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
Haug = kron(H, eye(K));  % augmented H for CPLEX

%% Formulate and solve the exact problem
[A, b] = getConstraints(N, K);

[x_cplex,fval_cplex,time_cplex, cplexStatus] = solverCPLEX(N, K, Haug, g, A, b);
fprintf ('\nsolverCPLEX: Solution status = %s \n', cplexStatus);
fprintf ('solverCPLEX: optimal value = %f \n', fval_cplex);
fprintf('time taken(s): %f \n\n', time_cplex);

if abs(fval_cplex - (trace(H*G*G')-trace(G'*G))) < 1e-5
    warning('CPLEX solution is the same as the unary terms')
end

X = reshape(x_cplex, K, N)';
label = X*(0:K-1)';

% The offset term for computing cost of the initial MRF
% f_MRF = fval + offset
offset = -(sum(H(:)) + sum(G(:)));
end

