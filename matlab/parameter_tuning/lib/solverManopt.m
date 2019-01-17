function [x, fval_manopt, fval_rounded, time_solver] = solverManopt(N, K,  Hcompact, Gcompact, problem, Y0, manopt_opts, SESync_opts)
% function computing the global minimum for multiclass segmentation using
% manopt
% Inputs:
% - N: number of superpixels
% - K: number of classes
% - Hcompact, Gcompact: energy function parameters trace(X'HX + GX')
% - (X is a N by K matrix, entry in X is either 0 or 1)
% - problem: problem setup (cost funciont, gradient, Hessian, etc.)
% - Y0: initial guess for manopot
% - manopt_opts: manopt solver options
% Date: 2/11/2018
% Authors: Luca Carlone, Siyi Hu

% initialize
maxIter = SESync_opts.maxIter;
minEigTol = SESync_opts.minEigTol;

time_solver = 0;
iter = 0;
if isempty(Y0)  % if Y0 is not specified, use a randomly generated initial guess
    Y0 = problem.M.rand();
end


%% RIEMANNIAN STAIRCASE (TODO: check second order critical point)
for r = K+1 : K+maxIter -1
    iter = iter + 1;
    fprintf('\nRIEMANNIAN STAIRCASE iteration %d (level r = %d):\n', iter, r);
    
    % solve using manopt
    [Yopt, fval_manopt, manopt_info, ~] = manoptsolve(problem, Y0, manopt_opts);
    time_solver = time_solver + manopt_info(end).time;
    
    % check optimzality
    fprintf('\nChecking second-order optimality...\n');
    Y = vertcat(Yopt.A, Yopt.B');
    eigVals = eig(Y*Y');
    eigVals = sort(eigVals, 'descend');
    if eigVals(r) < minEigTol
        fprintf('Found second-order critical point! (minimum eigenvalue = %g)\n', eigVals(r));
        break;
    else
        % Augment the dimensionality of the Stiefel manifolds in
        % preparation for the next iteration
        A = stiefelstackedfactory(N, 1, r+1); 
        B = stiefelfactory(r+1, K);           
        problem.M = productmanifold(struct('A', A, 'B', B));
        Y0 = problem.M.rand();
    end
end

%% check this Extract solution: compute X from Y
Z = Y*Y';
X_sdp = Z(1:N, N+1:N+K);
% % rank r = K approximation:
% [U,S,~] = svd(full(Z(1:N, 1:N)));
% r = K;
% X_sdp = U(:,1:r)*sqrt(S(1:r,1:r));  % estimated solution
fprintf ('mismatch full vs. rank K solution: %f\n', norm(X_sdp*X_sdp'-Z(1:N, 1:N)))

if sum(X_sdp(:)) < 0       % flip the sign if necessary                  
    X_sdp = -X_sdp;     %(X_sdp and -X_sdp are both valid solutions)
end

%% Extract solution: write x_sdp as a binary vector
% round result by picking the largest value for each pixel
x = zeros(N*K, 1);
[~, ind] = max(X_sdp, [], 2);       % find the index of the largest number in each row
ind = ind + (0:N-1)'*K;             % adjust index by row number
x(ind) = 1;                     

% compute feasibility gap:
X_rounded = reshape(x,[K,N])';
d = norm(X_sdp(:) - X_rounded(:),Inf);
fprintf('min=%g, max=%g, gap =%g\n',min(X_sdp(:)),max(X_sdp(:)),d)

% compute fval with the rounded solution
fval_rounded = trace(X_rounded'*Hcompact*X_rounded) + trace(Gcompact*X_rounded');
end