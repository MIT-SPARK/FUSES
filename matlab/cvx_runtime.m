% Script running image segmentation for cityscapes dataset to check cvx
% runtime

% Date: 9/8/2018
% Authors: Luca Carlone, Siyi Hu

clear all; close all; clc;
addpath('./lib')

%% Datasets / settings
useHDF5 = false;
nrTechniques = 2;
nrCoresStr = ''; %'taskset -c 0 '
datasetName = 'lindau';
datasetPath = '/mnt/dataset/Cityscapes'; % Siyi
if useHDF5
    % mrfFolderPath = '/home/luca/Desktop/dataset-Cityscapes-HDF5/'; % Luca ubuntu
    mrfFolderPath = '~/Desktop/data/dataset-Cityscapes-HDF5/'; % Luca mac
    % mrfFolderPath = '/home/siyi/dataset-Cityscapes-HDF5/'; % Siyi
    mrfFolderName = horzcat(datasetName,'-1000SP/');
    % mrfFolderName = horzcat(datasetName,'-1000SP-5classes/');
    % warning('using 5 class dataset')
end


%% Constants
frontendAppName = 'frontend-bonnet-hdf5'; % Cityscapes dataset based on bonnet

% parameter file
paramFileName = '../tests/data/fusesParameters.yaml';

% backend optimizers
fusesAppName = 'optimization-FUSES';
% text file containing the list of MRF files
tempFileName = 'tempNames.txt';

% Commands
frontendCommand = horzcat(nrCoresStr,' ../build/bin/%s ', datasetPath, ' ',datasetName, ' ../bonnet/frozen/ ', paramFileName);

if useHDF5
    backendCommand = horzcat(nrCoresStr,' ../build/bin/%s ', mrfFolderPath, mrfFolderName, '/%s ', mrfFolderPath, mrfFolderName, '/%s');
else
    backendCommand = horzcat(nrCoresStr,' ../build/bin/%s %s %s');
end
%% Runing front end
if useHDF5 % use precomputed MRF from Bonnet frontend
    disp('running script:')
else
    disp('running script with front end:')
    frontendCommand = sprintf(frontendCommand, frontendAppName);
    disp(' ')
    disp(frontendCommand)
    system(frontendCommand);
end

%% Getting file names from front end output
if useHDF5
    fileID = fopen(horzcat(mrfFolderPath, mrfFolderName, '/', tempFileName));
else
    fileID = fopen(tempFileName);
end
textFiles = textscan(fileID, '%s'); % 2 files per dataset: MRF and ground truth
nrFrames = length(textFiles{1})/2;
% parse MRF and ground truth
hdfFiles = cell(1, nrFrames);
labelGTFiles = cell(1, nrFrames);
for i = 1:nrFrames
    hdfFiles{i} = textFiles{1}{2*i-1};
    labelGTFiles{i} = textFiles{1}{2*i};
end
% instantiate exact solution using CPLEX
fval_exact = zeros(1, nrFrames);
labels_exact = cell(1, nrFrames);
offset = zeros(1, nrFrames); % offset wrt MRF

%% Runing all backend algorithms
for i = 1:nrFrames
    fprintf('=================== PROCESSING FRAME %d of %d ====================\n',i,nrFrames)
    hdfFileName = hdfFiles{i};
    labelGTFileName = labelGTFiles{i};
    
    % fuses
    fusesCommand = sprintf(backendCommand, fusesAppName, hdfFileName, labelGTFileName);
    disp(' ')
    disp(fusesCommand)
    system(fusesCommand);
    fprintf('')
    
    % exact solution using CPLEX
    if useHDF5
        [label, fval_cplex, offSet] = computeExact(horzcat(mrfFolderPath,mrfFolderName,hdfFileName));
        fval_exact(i) = fval_cplex;
        labels_exact{i} = label;
        offset(i) = offSet;
    else
        [label, fval_cplex, offSet] = computeExact(hdfFileName);
        fval_exact(i) = fval_cplex;
        labels_exact{i} = label;
        offset(i) = offSet;
        [x, fval_sdp, fval_rounded, time_solver, cvxStatus] = computeSDP(hdfFileName, false);
    end    
end
