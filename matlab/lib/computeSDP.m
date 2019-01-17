function [x, fval_sdp, fval_rounded, time_solver, cvxStatus] = computeSDP(filename, isRelaxed)
% function computing the global minimum for multiclass segmentation using
% SDP relaxation
% Inputs:
% - N: number of superpixels
% - K: number of classes
% - Haug, g: original energy function parameters x'Haugx + g'x
% - A, b: original equality constraint parameters Ax = b
% - isRelaxed: if set to true, the last equality constraint is relaxed  
% (x'Haugx + g'x) => (x_star'H_tilde x_star+g_tilde'x_star+offset)
% (x_star is the decision variable. It is a N*K by 1 vector with entries of -1 or 1)
% Date: 9/8/2018
% Authors: Luca Carlone, Siyi Hu

if nargin < 7 || ~isRelaxed
    disp('========== RUNNING: SDP (-1,+1) formulation ===================')
else
    disp('========== RUNNING: SDP (-1,+1) formulation without equality ==')
end

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

% convert parameters
% (x'Haug x + g'x) = (1/2(x_star+1)' Haug 1/2(x_star+11) + g' 1/2(x_star+1))
% = 1/4 x_star' Haug x_star + 1/2 (1' Haug + g') x_star + 1/4 1'Haug 1 + 1/2 g 1'
H_tilde = 1/4*Haug;
g_tilde = 1/2*(Haug*ones(N*K,1)) + 1/2*g;
offset = 1/4*sum(Haug(:)) + 1/2*sum(g);

% (A*(x_star+1)/2=b) => A*x_star = 2b-A*1 
Aeq = A;
beq = 2*b-A*ones(N*K,1);

% cvx setup
cvx_quiet(true)
cvx_solver mosek
cvx_begin %SDP
variable Z(N*K+1, N*K+1) symmetric;
minimize( trace(H_tilde*Z(1:N*K, 1:N*K)) + g_tilde'*Z(1:N*K, N*K+1) )
subject to
    Z == semidefinite(N*K+1);   % Z is semidefinite
    diag(Z) == ones(N*K+1,1);   % (Zii = 1) <= (xii^2 =1)
    
    % add additional constraints if not relaxed
    if nargin < 7 || ~isRelaxed
        Aeq*Z(1:N*K, N*K+1) == beq;    
    end

% solve SDP
timeSDP = tic;
cvx_end
time_solver = toc(timeSDP);
fval_sdp = cvx_optval+offset;   % add offset
cvxStatus = cvx_status;

%% check this
% [V, D] = eigs(Z, 1, 'largestabs');
% x_sdp = sqrt(D)*V(1:end-1);
% rank r=1 approximation:
[U,S,~] = svd(full(Z(1:N*K, 1:N*K)));
r = 1;
x_sdp = U(:,1:r)*sqrt(S(1:r,1:r));  % estimated solution
if sum(x_sdp) > 0  % flip the sign if necessary                 
    x_sdp = -x_sdp; 
end
fprintf ('mismatch full vs. rank 1 solution: %f\n', norm(x_sdp*x_sdp'-Z(1:N*K, 1:N*K)))

%% Extract solution: write x_sdp as a binary vector
% round result by picking the largest value for each pixel
x = zeros(N*K, 1);
for i = 1:N
    [~, ind] = max(x_sdp((i-1)*K+1:i*K));
    x((i-1)*K+ind) = 1; 
end

% compute feasibility gap:
x_rounded = 2*x-ones(N*K,1);
d = norm(x_sdp - x_rounded, Inf);
fprintf('min=%g,max=%g,gap=%g\n',min(x_sdp),max(x_sdp),d);
if norm(Aeq*x_rounded - beq,Inf)>1e-4
    norm(Aeq*x_rounded - beq,Inf)
    error('rounded solution is infeasible!') 
end

% compute fval with the rounded solution
fval_rounded = x_rounded'*H_tilde*x_rounded + g_tilde'*x_rounded+offset;

fprintf ('\nsolverSDP: Solution status = %s \n', cvxStatus);
fprintf ('solverSDP: optimal value = %f \n', fval_sdp);
fprintf('time taken(s): %f \n\n', time_solver);
end