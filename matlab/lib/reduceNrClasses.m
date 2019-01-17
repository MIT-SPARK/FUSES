%% rewrite hdf5 file after reducing number of labels in CRF
error('THIS FILE IS DEPRECATED - the reduction of the CRF classes is implemented in CRF.h')

clear all
close all
clc

h5disp(filename)

filename = '/home/luca/Desktop/dataset-Cityscapes-HDF5/toyCRFfile.hdf5'

data = h5read(filename,'/gm/numbers-of-states');
N = length(data);
K = data(1);

% get data matrices
g = h5read(filename,'/gm/function-id-16000/values');
G = reshape(g, K, N)';

indices = h5read(filename,'/gm/factors');
% % first 4*N entries are for the unary terms
% indices = indices(4*N+1:end); % indices for binary terms
% data = h5rNead(filename,'/gm/function-id-16006/values');

7,5,
0,0,
1,0,
2,1,
3,0,
4,2,
0,1,0.25
0,3,0.15
1,2,0.225
1,3,0.2
2,3,0.175
2,4,0.1
3,4,0.05