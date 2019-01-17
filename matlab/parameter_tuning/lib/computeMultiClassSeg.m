function [cplexOut, sdp4Out, manoptOut] = computeMultiClassSeg(imageRGBName, imageLabelsName, nrSP, dataMatrices, figureOn)
% function solving image segmentation problem using CPLEX, SDP-compact and
% Riemannian staircase
% Date: 6/19/2018
% Authors: Siyi Hu

%% Inputs
% constants
SESync_opts.maxIter = 5;        % max # of iterations for Riemanian staircase
SESync_opts.minEigTol = 1e-4;   % eigenvalue tolerance in oprimality verification
manopt_opts.solver = @trustregions;
manopt_opts.tolgradnorm = 1e-2; % Stopping tolerance for norm of Riemannian gradient

% data matrices
Hcompact = dataMatrices.Hcompact;
Gcompact = dataMatrices.Gcompact;
   
%% Superpixels
imageRGB = imread(imageRGBName);
imageLabels = imread(imageLabelsName);

% get superpixels
[L, N] = superpixels(imageRGB, nrSP); 
spIndices = label2idx(L);
if figureOn
    figure
    BW = boundarymask(L);
    imshow(imoverlay(imageRGB,BW,'cyan'),'InitialMagnification',100)
end

fprintf('%i super pixels computed for image %s.\n', N, imageRGBName)

%% Formulate optimization: E = x'Hx + g'x, Ax = b (assume x is either 1 or -1)
% number of classes
K = double(max(max(imageLabels))) + 1;    

% binary term
% H is a N by N matrix for the second SDP relaxation (X is a matrx)
Haug = kron(Hcompact, eye(K));  % augmented H for CPLEX

% unary term
% G is a N by K matrix for the second SDP relaxation (X is a matrx)
g = reshape(Gcompact', [], 1);

% equality constraint
[A, b] = getConstraints(N, K);


%% Solve optimization 
% Although formulations are different, the final result x is a binary 
% vector for all three methods

%% CPLEX
[x_cplex,fval_cplex,time_cplex, cplexStatus] = solverCPLEX(N, K, Haug, g, A, b);
fprintf ('\nsolverCPLEX: Solution status = %s \n', cplexStatus);
fprintf ('solverCPLEX: optimal value = %f \n', fval_cplex);
fprintf('time taken(s): %f \n\n', time_cplex);

% output struct
cplexOut.x = x_cplex;
cplexOut.fval = fval_cplex;
cplexOut.time = time_cplex;
    
%% SDP4 (matrix X - relaxed)
[x_sdp4, fval_sdp4, fval_rounded4, time_sdp4, cvxStatus] = solverSDPcompact(N, K, Hcompact, Gcompact, true);
fprintf ('solverSDP_compact_relaxed: Solution status = %s \n', cvxStatus);
fprintf ('solverSDP_compact_relaxed: optimal value = %f \n', fval_sdp4);
fprintf ('solverSDP_compact_relaxed: optimal value (rounded) = %f \n', fval_rounded4);
fprintf('time taken(s): %f \n\n', time_sdp4);

% output struct
sdp4Out.x = x_sdp4;
sdp4Out.fval = fval_sdp4;
sdp4Out.fval_rounded = fval_rounded4;
sdp4Out.time = time_sdp4;

%% Manopt
% setup the problem - manifold (Y = [A; B'])
A = stiefelstackedfactory(N, 1, K); % X
B = stiefelfactory(K, K);           % I_K
problem.M = productmanifold(struct('A', A, 'B', B));

% setup the problem - objective function, Euclidean gradient, Euclidean
% Hessian vector product
problem.cost = @(Y) trace(Hcompact*Y.A*Y.A')+trace(Gcompact*Y.B'*Y.A');
problem.egrad = @(Y) struct('A', 2*Hcompact*Y.A + Gcompact*Y.B', 'B', Y.A'*Gcompact);
problem.ehess = @(Y, Ydot) struct('A', 2*Hcompact*Ydot.A + Gcompact*Ydot.B', 'B', Ydot.A'*Gcompact);

 % solve the problem using manopt and compute rounded solution
[x_manopt, fval_manopt, fval_rounded, time_manopt] = solverManopt(N, K,  Hcompact, Gcompact, problem, [], manopt_opts, SESync_opts);
fprintf ('solverManopt: optimal value = %f \n', fval_manopt);
fprintf ('solverManopt: optimal value (rounded) = %f \n', fval_rounded);
fprintf('time taken(s): %f \n\n', time_manopt);

fprintf('Gap between Manopt and SDP4: %f\n', fval_manopt-fval_sdp4)
fprintf('After rounding, this gap becomes: %f\n\n', fval_rounded-fval_rounded4)

% output struct
manoptOut.x = x_manopt;
manoptOut.fval = fval_manopt;
manoptOut.fval_rounded = fval_rounded;
manoptOut.time = time_manopt;

%% Plot results
if figureOn
    % plot ground truth
%     color = [0 0 0; 255 0 0; 0 255 0; 255 255 0; 0 0 255]; % this is hard-coded - change later
%     plotGroundTruth(imageRGB, imageLabels, color)

    % get segmented images
    resultsName = ' (';
    
    imageSeg0 = getSegmentedImage(x_cplex, K, N, imageRGB, imageLabels, spIndices);
    resultsName = horzcat(resultsName, 'CPLEX, ');

    imageSeg4 = getSegmentedImage(x_sdp4, K, N, imageRGB, imageLabels, spIndices);
    resultsName = horzcat(resultsName, 'SDP matrix X - relaxed, ');

    imageSeg5 = getSegmentedImage(x_manopt, K, N, imageRGB, imageLabels, spIndices);
    resultsName = horzcat(resultsName, 'Manopt, ');

    resultsName(end-1) = ')';

    % plot segmented image
    % turn off the warning about image being too big for the screen
    id = 'images:initSize:adjustingMag';
    warning('off',id)   
    for k = 1:K
        figure
        imageSeg = horzcat(imageSeg0{k}, imageSeg4{k}, imageSeg5{k});
        imshow(imageSeg)
        title(['Region ' num2str(k) resultsName])
    end
end
end
