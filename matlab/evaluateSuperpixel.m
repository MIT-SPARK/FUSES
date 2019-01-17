% Script running image segmentation for cityscapes dataset using four
% different algorithms: Fuses, alpha-expansion, loopy-belief-propagation
% and tree reweighted message passing (TRW-S)

% Date: 6/13/2018
% Authors: Luca Carlone, Siyi Hu

clear all; close all; clc;
addpath('./lib')

%% Datasets / settings
datasetName = 'lindau';
datasetPath = '/mnt/dataset/Cityscapes'; % Siyi

%% Constants
appName = 'superpixel-test'; 

% parameter file
paramFileName = '../tests/data/fusesParameters.yaml';

% text file containing the list of label files' and pixel level info
tempFileName = 'tempNames-SP.txt';

% csv file storing sfe initialization time
timeFileName = 'initializationTime.csv';

%% Running command
command = horzcat('../build/bin/%s ', datasetPath, ' ',datasetName, ' ', paramFileName);
command = sprintf(command, appName);
disp(command)
system(command);

%% Getting file names from front end output
fileID = fopen(tempFileName);
textFiles = textscan(fileID, '%s'); % 2 files per frame
nrFrames = length(textFiles{1})/2;

labelGTFiles = cell(1, nrFrames);
pixelGTFiles = cell(1, nrFrames);
for i = 1:nrFrames
    labelGTFiles{i} = textFiles{1}{2*i-1};
    pixelGTFiles{i} = textFiles{1}{2*i};
end

%% Parsing results
mAccuracyPerc = zeros(1, nrFrames); % mAccuracy (i.e. PA)
mIoUPerc = zeros(1, nrFrames); % mIoU
nrNodes = zeros(1, nrFrames);
for i = 1:nrFrames
    labelGTFileName = labelGTFiles{i};
    pixelGTFileName = pixelGTFiles{i};
    
    labelGT = csvread(labelGTFileName);
    labelGT_table = csvread(pixelGTFileName);
    
    nrNodes(1, i) = length(labelGT);
    
    [PA, ~, MIoU, ~] = computeAccuracy(labelGT_table, labelGT);
    mAccuracyPerc(1, i) = PA;
    mIoUPerc(1, i) = MIoU;
end

%% Dispaly statistics
nrNodes_mean = mean(nrNodes);
nrNodes_std = std(nrNodes);
disp('Average number of nodes with standard deviation')
disp(horzcat(nrNodes_mean, nrNodes_std))

mAccuracyPerc_mean = mean(mAccuracyPerc);
mAccuracyPerc_std = std(mAccuracyPerc);
disp('Pixel level accuracy with standard deviation - mAccuracy')
disp(horzcat(mAccuracyPerc_mean, mAccuracyPerc_std))

mIoUPerc_mean = mean(mIoUPerc);
mIoUPerc_std = std(mIoUPerc);
disp('Pixel level accuracy with standard deviation - mIoU')
disp(horzcat(mIoUPerc_mean, mIoUPerc_std))

timeData = csvread(timeFileName);
time_mean = mean(timeData);
time_std = std(timeData);
disp('Initialization time with standard deviation (s)')
disp(horzcat(time_mean, time_std))