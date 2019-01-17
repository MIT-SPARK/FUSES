function [x, fval_sdp, fval_rounded, time_solver, cvxStatus] = solverSDPcompact(N, K, Hcompact, Gcompact, isRelaxed)
% function computing the global minimum for multiclass segmentation using
% SDP relaxation - compact
% Inputs:
% - N: number of superpixels
% - K: number of classes
% - Hcompact, Gcompact: energy function parameters trace(X'HX + GX')
% - (X is a N by K matrix, entry in X is either 0 or 1)
% - isRelaxed: if set to true, the last two constraints are relaxed  
% Date: 1/24/2018
% Authors: Luca Carlone, Siyi Hu

% trace(C*Z) <= trace(Hcompact*Z(1:N, 1:N)+Gcompact*Z(N+1:N+K, 1:N)
C = spalloc(N+K, N+K, nnz(Hcompact)+nnz(Gcompact));
C(1:N, 1:N) = Hcompact;
C(1:N, N+1:end) = Gcompact;

% cvx setup
cvx_quiet(true)
cvx_solver mosek
cvx_begin %SDP
variable Z(N+K, N+K) symmetric;
minimize( trace(C*Z) )
subject to
    Z == semidefinite(N+K);             % Z is semidefinite
    diag(Z) == ones(N+K,1);             % sum_k(xik^2) = 1;   
    Z(N+1:N+K, N+1:N+K) == speye(K);    % lower right part of Z is I
    
    % add additional constraints if not relaxed
    if nargin < 5 || ~isRelaxed
        Z(1:N, N+1:N+K)*ones(K,1) == ones(N,1); % sum_k(xik) = 1;     
        Z >= 0;                                 % xik >= 0
    end

% solve SDP
timeSDP = tic;
cvx_end
time_solver = toc(timeSDP);
fval_sdp = cvx_optval;
cvxStatus = cvx_status;

% get eigenvalues
e = eig(Z);
e = sort(e,'descend');

%% check this
X_sdp = Z(1:N, N+1:N+K);
% % rank r = K approximation:
% [U,S,~] = svd(full(Z(1:N, 1:N)));
% r = K;
% X_sdp = U(:,1:r)*sqrt(S(1:r,1:r));  % estimated solution
fprintf('mismatch full vs. rank K solution: %f\n', norm(X_sdp*X_sdp'-Z(1:N, 1:N)))

if sum(X_sdp(:)) < 0       % flip the sign if necessary                  
    X_sdp = -X_sdp;     %(X_sdp and -X_sdp are both valid solutions)
end

%% Extract solution: write X_sdp as a binary vector
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