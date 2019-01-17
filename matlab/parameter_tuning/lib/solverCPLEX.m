function [x,fval,time_solver, status] = solverCPLEX(N, K, Haug, g, Aeq, beq)
% function computing the global minimum for multiclass segmentation
% using CPLEX
% Inputs:
% - N: number of superpixels
% - K: number of classes
% - H, g: energy function parameters x'Hx + g'x
% - Aeq, beq: equality constraint parameters Aeq x = beq
% - (entry in x is either 0 or 1)
% Date: 1/23/2018
% Authors: Luca Carlone, Siyi Hu

% CPLEX setup
Aineq = []; bineq = [];
sostype = []; sosind = []; soswt = []; lb = []; ub = []; 
options = cplexoptimset;
options.Display = 'off';
% Set ctype(j) to 'B', 'I','C', 'S', or 'N' to 
% indicate that x(j) should be binary, general integer, 
% continuous, semi-continuous or semi-integer (respectively).
ctype = blanks(N*K);
for i=1:N*K
   ctype(i) = 'B'; % horzcat(ctype,'B'); 
end

% CPLEX PROBLEM:
% (cplexmiqp solves minimization): min 1/2*x'*H*x + f*x
% OUR PROBLEM: x'Hx+g'x
Hcplex = 2*Haug;
fcplex = g';

try
    timeCplex = tic;
    [x, fval, exitflag, output] = ... 
        cplexmiqp(Hcplex,fcplex,Aineq,bineq,Aeq,beq,...
        sostype,sosind,soswt,lb,ub,ctype, [], options);
    time_solver = toc(timeCplex);
    status = output.cplexstatusstring;
catch 
    disp('Error when running cplex - is it installed correctly?')
    x = nan(N,1);
    fval = nan;
    time_solver = nan;
end
end