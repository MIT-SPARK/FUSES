% Script running frontend on cityscapes dataset to compute average variance
% of color difference (between neighboring superpixels)

% Date: 7/5/2018
% Authors: Luca Carlone, Siyi Hu

% clear all; close all; %clc;
% addpath('./lib')

%% Datasets / settings
datasetName = 'all';
datasetPath = '/mnt/dataset/Cityscapes'; % Siyi

%% Constants
% parameter file
paramFileName = '../tests/data/fusesParameters.yaml';

% backend optimizers
fittingAppName = 'fuses-fittingBeta';

% csv file containing the average color variance for each frame
dataFileName = 'colorVariances.csv';

%% Running command
command = horzcat('../build/bin/%s ', datasetPath, ' ',datasetName, ' ', paramFileName);
command = sprintf(command, fittingAppName);
disp(command)
system(command);

%% Compute statistics
colorVar = csvread(dataFileName);
beta = 1 / (2*mean(colorVar));
fprintf('Average color variance = %d, beta fitted = %d.\n', mean(colorVar), beta);

% krefeld
% Average color variance = 3767.66, beta fitted = 0.000132708

% aachen
% Average color variance = 2087.91, beta fitted = 0.000239474

% bremen
% Average color variance = 2870.28, beta fitted = 0.000174199

% darmstadt
% Average color variance = 3846.54, beta fitted = 0.000129987

% lindau
% Average color variance = 1987.14, beta fitted = 0.000251617
% Average gray-scale variance = 671.37, beta fitted = 0.0007447499

% munster
% Average color variance = 2173.31, beta fitted = 0.000230064

% frankfurt
% Average color variance = 3458.39, beta fitted = 0.000144576
% Average gray-scale variance = 1165.87, beta fitted = 0.0004288657
