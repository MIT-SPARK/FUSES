% Script finding superpixel accuracy

% Date: 12/8/2018
% Authors: Luca Carlone, Siyi Hu

% clear all; close all; clc;
addpath('./lib')

%% Datasets / settings
datasetPath = '/mnt/dataset/Cityscapes'; % Siyi
resultPath = './results-perfect2000';

%% Constants
hdf5modifyAppName = 'frontend-rewrite-hdf5';

% bonnet accuracy
bonnetAccuracyAppName = 'frontend-bonnet-computeAccuracy'; 

% text file containing the list of MRF files
tempFileName = 'tempNames.txt';
tempFilePath = horzcat(resultPath, '/', tempFileName);

%% Commands
hdf5modifyCommand = horzcat('../build/bin/', hdf5modifyAppName, ' %s %s %f %f');

bonnetAccuracyCommand = horzcat('../build/bin/', bonnetAccuracyAppName, ' ', datasetPath, ' %s');

%% Getting file names from front end output
fileID = fopen(tempFilePath);
textFiles = textscan(fileID, '%s'); % 2 files per dataset: MRF and ground truth

% parse MRF and ground truth
nrFrames = length(textFiles{1})/2;
noWeight_hdfFiles = cell(1, nrFrames); % stores binary factor as lambda1 = 0, lambda2 = 1
labelGTFiles = cell(1, nrFrames);
for i = 1:nrFrames
    noWeight_hdfFiles{i} = textFiles{1}{2*i-1};
    labelGTFiles{i} = textFiles{1}{2*i};
end

%% log accuracy
mAccuracyBonnet = zeros(1, nrFrames);
mIoUBonnet = zeros(1, nrFrames);

fval_exact = zeros(1, nrFrames);
labels_exact = cell(1, nrFrames);

for i = 1:nrFrames
    fprintf('=================== PROCESSING FRAME %d of %d ====================\n',i,nrFrames)
    noWeight_hdfFileName = noWeight_hdfFiles{i};
    
    % bonnet accuracy
    bonnetImageName = horzcat(noWeight_hdfFileName(1:end-3), '_bonnet.png');
    command = sprintf(bonnetAccuracyCommand, bonnetImageName);
    disp(' ')
    disp(command)
    system(command);
    resultFileName = horzcat(noWeight_hdfFileName(1:end-3), '_bonnetAccuracy.csv');
    P = csvread(resultFileName);
    [PA, ~, MIoU, ~] = computeAccuracyfromMatrix(P);
    mAccuracyBonnet(i) = PA;
    mIoUBonnet(i) = MIoU;
    
    % Modify hdf5 file to remove binary terms
    hdfFileName = horzcat(noWeight_hdfFileName(1:end-3), '_m.h5');
    modifyCommand = sprintf(hdf5modifyCommand, noWeight_hdfFileName, hdfFileName, 0.0, 0.0);
    disp(' ')
    disp(modifyCommand)
    system(modifyCommand);
    fprintf('')
    
    % exact solution using CPLEX
    [label, fval_cplex] = computeExact(hdfFileName);
    fval_exact(i) = fval_cplex;
    labels_exact{i} = label;
end

%% Parsing results
nrNodes = zeros(1, nrFrames); % counts nr of nodes for each dataset
time = zeros(1, nrFrames);
mAccuracyCplex = zeros(1, nrFrames);
mIoUCplex = zeros(1, nrFrames);
for i = 1:nrFrames
    fprintf('Parsing results for frame %d\n',i)
    noWeight_hdfFileName = noWeight_hdfFiles{i};
    hdfFileName = horzcat(noWeight_hdfFileName(1:end-3), '_m.h5');
    labelGTFileName = labelGTFiles{i};
      
    % pixel-level accuracy
    pixelLabelGT = horzcat(noWeight_hdfFileName(1:end-3),'_pixelLabels.csv');
    labelGT_table = csvread(pixelLabelGT);
    [PA, ~, MIoU, ~] = computeAccuracy(labelGT_table, labels_exact{i});
    mAccuracyCplex(1, i) = PA;
    mIoUCplex(1, i) = MIoU;
end
        

%% display statistics
disp('Bonnet accuracy')
disp([mean(mIoUBonnet) std(mIoUBonnet)])

disp('Unary accuracy')
disp([mean(mIoUCplex) std(mIoUCplex)])
