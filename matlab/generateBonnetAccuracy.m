% Script computing accuracy statistics for bonnet
% Note that this script can only be used when bonnet .png images as well as
% a text file (tempFileName) storing hdf5 and label file names are already 
% obtained. These files have to be in the same folder as this script

% Date: 7/18/2018
% Authors: Luca Carlone, Siyi Hu

% clear all; close all; clc;
addpath('./lib')

%% Datasets / settings
% datasetName = 'lindau';
datasetPath = '/mnt/dataset/Cityscapes'; % Siyi
resultPath = './results-fitting';

%% Constants
appName = 'frontend-bonnet-computeAccuracy'; 
% text file containing the list of MRF files
tempFileName = 'tempNames-fitting.txt';

% Command
appCommand = horzcat('../build/bin/', appName, ' ', datasetPath, ' %s');

%% Parse MRF and ground truth
fileID = fopen(horzcat(resultPath, '/', tempFileName));
textFiles = textscan(fileID, '%s'); % 2 files per dataset: MRF and ground truth
nrFrames = length(textFiles{1})/2;
hdfFiles = cell(1, nrFrames);
labelGTFiles = cell(1, nrFrames);
for i = 1:nrFrames
    hdfFiles{i} = textFiles{1}{2*i-1};
    labelGTFiles{i} = textFiles{1}{2*i};
end
%% Run script for all _bonnet.png images
disp('running script to compute Bonnet accuracy statistics:')
nrFrames = 100;
PAList = zeros(1, nrFrames);
MPAList = zeros(1, nrFrames);
MIoUList = zeros(1, nrFrames);
FWIoUList = zeros(1, nrFrames);
PAList_GT = zeros(1, nrFrames);
MIoUList_GT = zeros(1, nrFrames);
for i = 1:nrFrames
    hdfFileName = hdfFiles{i};
    labelGTFileName = labelGTFiles{i};
    hdfFileNameNoExtension = hdfFileName(1:end-3);
    
    bonnetImageName = horzcat(hdfFileNameNoExtension, '_bonnet.png');
    command = sprintf(appCommand, bonnetImageName);
    disp(' ')
    disp(command)
    system(command);
    
    % bonnet accuracy
    resultFileName = horzcat(hdfFileNameNoExtension, '_bonnetAccuracy.csv');
    P = csvread(resultFileName);
    [PA, MPA, MIoU, FWIoU] = computeAccuracyfromMatrix(P);
    PAList(i) = PA;
    MPAList(i) = MPA;
    MIoUList(i) = MIoU;
    FWIoUList(i) = FWIoU;
    
    % GT accuracy
    pixelLabelGT = horzcat(hdfFileNameNoExtension,'_pixelLabels.csv');
    labelGT_table = csvread(pixelLabelGT);
    labelGT = csvread(labelGTFileName);
    [PA, ~, MIoU, ~] = computeAccuracy(labelGT_table, labelGT);
    PAList_GT(i) = PA;
    MIoUList_GT(i) = MIoU;
    
end


%% Compute statistics
PA_mean = mean(PAList);
PA_std = std(PAList);
disp('PA mean and standard deviation')
disp(horzcat(PA_mean, PA_std))

MPA_mean = mean(MPAList);
MPA_std = std(MPAList);
disp('MPA mean and standard deviation')
disp(horzcat(MPA_mean, MPA_std))

MIoU_mean = mean(MIoUList);
MIoU_std = std(MIoUList);
disp('MIoU mean and standard deviation')
disp(horzcat(MIoU_mean, MIoU_std))

FWIoU_mean = mean(FWIoUList);
FWIoU_std = std(FWIoUList);
disp('FWIoU mean and standard deviation')
disp(horzcat(FWIoU_mean, FWIoU_std))

%% plot figure
figure
plot(1:nrFrames, MIoUList, 'b-', 1:nrFrames, MIoU_mean*ones(1, nrFrames), 'r--')
xlabel('image index')
ylabel('mIoU % accuracy')