% Script running fitting lambda1 and lambda2 for cityscapes dataset

% Date: 7/5/2018
% Authors: Luca Carlone, Siyi Hu

clear all; close all; clc;
addpath('./lib')

%% Datasets / settings
useNewImages = false;
useCityscape = true; % run bonnet on cityscapes if true otherwise run DAVIS
findBonnetAccuracy = true; % log bonnet accuracy
if useCityscape
	datasetName = 'all';
	datasetPath = '/mnt/dataset/Cityscapes'; % Siyi
else
	datasetName = 'train';
    datasetPath = '/mnt/dataset/DAVIS';  % Siyi
%     datasetPath = '/home/luca/Desktop/DAVIS'; % Luca
    % mrfFolderPath = '/home/luca/Desktop/dataset-Cityscapes-HDF5/';
    % mrfFolderName = horzcat(datasetName,'-1000SP/');
end
resultPath = './results';

lambda1List = 0.00;
lambda2List = 0.06;
% lambda1List = -0.04:0.01:0.02;
% lambda2List = 0.04:0.01:0.10;

resultsfileID = fopen('fitting_results_noalpha_.txt', 'w');

%% Constants
if useCityscape
    frontendAppName = 'frontend-bonnet-hdf5-fitting'; % Cityscapes dataset based on bonnet
    hdf5modifyAppName = 'frontend-rewrite-hdf5';
else
%     frontendAppName = 'frontend-DAVIS-hdf5'; % DAVIS dataset based on ground truth
    error('Not implemented yet');
end
% parameter file
paramFileName = '../tests/data/fusesParameters_512_1000.yaml';

% backend optimizers
fusesAppName = 'optimization-FUSES';

% bonnet accuracy
bonnetAccuracyAppName = 'frontend-bonnet-computeAccuracy'; 

% text file containing the list of MRF files
tempFileName = 'tempNames.txt';
tempFilePath = horzcat(resultPath, '/', tempFileName);

%% Commands
if useCityscape
    frontendCommand = horzcat('../build/bin/%s ', datasetPath, ' ',datasetName, ' ../bonnet/frozen_512/ ', paramFileName);
else
%     frontendCommand = horzcat('taskset -c 0 ../build/bin/%s ', datasetPath, ' ',datasetName, ' ', paramFileName);
    error('Not implemented yet');
end

hdf5modifyCommand = horzcat('../build/bin/', hdf5modifyAppName, ' %s %s %f %f');

backendCommand = horzcat('taskset -c 0 ../build/bin/%s %s %s');

bonnetAccuracyCommand = horzcat('../build/bin/', bonnetAccuracyAppName, ' ', datasetPath, ' %s');

%% Process all images
if useNewImages
    disp('running script with front end:')
    command = sprintf(frontendCommand, frontendAppName);
    disp(' ')
    disp(command)
    system(command);
else
    disp('running script with existing frontend files:')
end

%% Getting file names from front end output
fileID = fopen(tempFilePath);
textFiles = textscan(fileID, '%s'); % 2 files per dataset: MRF and ground truth

% parse MRF and ground truth
nrFrames = length(textFiles{1})/2;
% nrFrames = 100;
noWeight_hdfFiles = cell(1, nrFrames); % stores binary factor as lambda1 = 0, lambda2 = 1
labelGTFiles = cell(1, nrFrames);
for i = 1:nrFrames
    noWeight_hdfFiles{i} = textFiles{1}{2*i-1};
    labelGTFiles{i} = textFiles{1}{2*i};
end

%% log bonnet accuracy
mAccuracyBonnet = zeros(1, nrFrames);
mIoUBonnet = zeros(1, nrFrames);
if findBonnetAccuracy
    % bonnet accuracy
    for i = 1:nrFrames
        noWeight_hdfFileName = noWeight_hdfFiles{i};
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
    end
end
        
%% Double loop for fitting lambda
% log data (average pecentage across all frames)
fusesGap = zeros(length(lambda1List), length(lambda2List));
fusesGap_rounding = zeros(length(lambda1List), length(lambda2List));
fusesSuboptLabels = zeros(length(lambda1List), length(lambda2List));
cplexAccuracy = zeros(length(lambda1List), length(lambda2List));
fuses_mAccuracy = zeros(length(lambda1List), length(lambda2List));
fuses_mIoU = zeros(length(lambda1List), length(lambda2List));
cplex_mAccuracy = zeros(length(lambda1List), length(lambda2List));
cplex_mIoU = zeros(length(lambda1List), length(lambda2List));
% log time
fusesTime = zeros(length(lambda1List), length(lambda2List));

for l1 = 1:length(lambda1List)
    for l2 = 1:length(lambda2List)
%     for l2 = l1+(1:3)
        lambda1 = lambda1List(l1);
        lambda2 = lambda2List(l2);
%         lambda2 = 0.06-lambda1;
        fprintf('=================== lambda 1 = %f, lambda 2 = %f ====================\n',lambda1,lambda2)
        
        if(nrFrames ~= length(textFiles{1})/2) warning('nrFrames > size of datasets: not running all frames'); end
        
        % instantiate exact solution using CPLEX 
        fval_exact = zeros(1, nrFrames);
        labels_exact = cell(1, nrFrames);

        %% Runing all backend algorithms
        for i = 1:nrFrames
            fprintf('=================== PROCESSING FRAME %d of %d ====================\n',i,nrFrames)
            noWeight_hdfFileName = noWeight_hdfFiles{i};
            labelGTFileName = labelGTFiles{i};
            
            % Modify hdf5 file
            hdfFileName = horzcat(noWeight_hdfFileName(1:end-3), '_m.h5');
            modifyCommand = sprintf(hdf5modifyCommand, noWeight_hdfFileName, hdfFileName, lambda1, lambda2);
            disp(' ')
            disp(modifyCommand)
            system(modifyCommand);
            fprintf('')
            
            % fuses
            fusesCommand = sprintf(backendCommand, fusesAppName, hdfFileName, labelGTFileName);
            disp(' ')
            disp(fusesCommand)
            system(fusesCommand);
            fprintf('')

            % exact solution using CPLEX
            [label, fval_cplex] = computeExact(hdfFileName);
            fval_exact(i) = fval_cplex;
            labels_exact{i} = label; 
        end
        
        %% Parsing results
        nrNodes = zeros(1, nrFrames); % counts nr of nodes for each dataset
        nrSuboptLabels = zeros(1, nrFrames);  % number against CPLEX labels
        time = zeros(1, nrFrames);
        fvalFuses = zeros(1, nrFrames);
        fvalFuses_rounding = zeros(1, nrFrames);
        accuracyCplex = zeros(1, nrFrames);
        mAccuracyFuses = zeros(1, nrFrames);
        mIoUFuses = zeros(1, nrFrames);
        mAccuracyCplex = zeros(1, nrFrames);
        mIoUCplex = zeros(1, nrFrames);
        for i = 1:nrFrames
            fprintf('Parsing results for frame %d\n',i)
            noWeight_hdfFileName = noWeight_hdfFiles{i};
            hdfFileName = horzcat(noWeight_hdfFileName(1:end-3), '_m.h5');
            labelGTFileName = labelGTFiles{i};            
            hdfFileNameNoExtension = noWeight_hdfFileName(1:end-3);
            
            % read in results
            fusesDataFile = horzcat(hdfFileNameNoExtension,'_m_FUSES.csv');
            % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
            fusesData_overall = csvread(fusesDataFile, 0, 1, [0,1,4,1]);
            % store number of nodes:
            nrNodes(i) = fusesData_overall(2);
            fusesLabel = csvread(fusesDataFile, 5, 1, [5,1,5,nrNodes(i)])'; % read labels
            fusesIterationsData = csvread(fusesDataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]
            
            % compute accuracy for cplex solution against ground-truth
            labelGT = csvread(labelGTFileName);
            accuracyCplex(1, i) = (1-nnz(labels_exact{i} - labelGT)/nrNodes(i)) * 100;
            
            % pixel-level accuracy
            pixelLabelGT = horzcat(hdfFileNameNoExtension,'_pixelLabels.csv');
            labelGT_table = csvread(pixelLabelGT);
            [PA, ~, MIoU, ~] = computeAccuracy(labelGT_table, fusesLabel);
            mAccuracyFuses(1, i) = PA;
            mIoUFuses(1, i) = MIoU;
            
            [PA, ~, MIoU, ~] = computeAccuracy(labelGT_table, labels_exact{i});
            mAccuracyCplex(1, i) = PA;
            mIoUCplex(1, i) = MIoU;
            
            % compute statistics for each frame
            nrSuboptLabels(1, i) = nnz(labels_exact{i} - fusesLabel);
            time(1, i) = fusesIterationsData(end, 3);
            fvalFuses(1, i) = fusesIterationsData(end, 2);
            fvalFuses_rounding(1, i) = fusesIterationsData(end, 4);
        end
        
%         % compute statistics for each lambda1, lambda2 combination
        fusesGap(l1, l2) = mean((fvalFuses - fval_exact)./fval_exact) * 100;
        fusesGap_rounding(l1, l2) = mean((fvalFuses_rounding - fval_exact)./fval_exact) * 100;
        fusesSuboptLabels(l1, l2) = mean(nrSuboptLabels./nrNodes) * 100;
        fusesTime(l1, l2) = mean(time);
        cplexAccuracy(l1, l2) = mean(accuracyCplex);
        fuses_mAccuracy(l1, l2) = mean(mAccuracyFuses);
        fuses_mIoU(l1, l2) = mean(mIoUFuses);
        cplex_mAccuracy(l1, l2) = mean(mAccuracyCplex);
        cplex_mIoU(l1, l2) = mean(mIoUCplex);
        
        fprintf(resultsfileID, horzcat('lambda1 = %f lambda2 = %f, fusesSuboptLabels = %f, ', ...
            'fusesTime = %f, cplex_mAccuracy = %f, cplex_mIoU = %f, fuses_mAccuracy = %f, fuses_mIoU = %f\n'), ...
            lambda1, lambda2, mean(nrSuboptLabels./nrNodes)*100, mean(time), mean(mAccuracyCplex), ...
            mean(mIoUCplex), mean(mAccuracyFuses), mean(mIoUFuses));
        
%         if findBonnetAccuracy
%             figure
%             plot(1:nrFrames, mIoUFuses, 'b-', 1:nrFrames, mIoUCplex, 'r-', ...
%                 1:nrFrames, mIoUBonnet, 'g--')
%             xlabel('image index')
%             ylabel('mIoU % accuracy')
%             legend('fuses', 'cplex', 'bonnet')
%         end
            
    end
end

%% display statistics
figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',fusesGap)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('Fuses percent gap before rounding')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',abs(fusesGap_rounding))
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('Fuses percent gap after rounding')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',fusesSuboptLabels)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('Fuses percent of suboptimal labels')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',fusesTime)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('Fuses runtime')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',cplexAccuracy)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('CPLEX accuracy')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',cplex_mAccuracy)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('CPLEX mAccuracy')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',cplex_mIoU)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('CPLEX mIoU')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',fuses_mAccuracy)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('FUSES mAccuracy')

figure
imagesc('XData',lambda2List,'YData',lambda1List,'CData',fuses_mIoU)
colorbar
xlabel('lambda2')
ylabel('lambda1')
title('FUSES mIoU')

save('results-data','lambda1List','lambda2List', 'fusesGap', 'fusesGap_rounding', ...
    'fusesSuboptLabels', 'fusesTime', 'fuses_mAccuracy', 'fuses_mIoU', ...
    'cplexAccuracy', 'cplex_mAccuracy', 'cplex_mIoU')
% fprintf('Bonnet label percent accuracy: %.2f +/- %.2f %%\n', ...
%     mean(BonnetAccuracy), std(BonnetAccuracy))