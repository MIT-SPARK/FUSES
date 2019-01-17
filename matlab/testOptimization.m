% Script running image segmentation for cityscapes dataset using four
% different algorithms: Fuses, alpha-expansion, loopy-belief-propagation
% and tree reweighted message passing (TRW-S)

% Date: 6/13/2018
% Authors: Luca Carlone, Siyi Hu

clear all; close all; clc;
addpath('./lib')

%% Datasets / settings
useCityscape = true; % run bonnet on cityscapes if true otherwise run DAVIS
useHDF5 = false;
saveEps = false;
nrTechniques = 7;
nrCoresStr = ''; %'taskset -c 0 '
if useCityscape
    datasetName = 'frankfurt';
    datasetPath = '/mnt/dataset/Cityscapes'; % Siyi
    if useHDF5
        % mrfFolderPath = '/home/luca/Desktop/dataset-Cityscapes-HDF5/'; % Luca ubuntu
        mrfFolderPath = '~/Desktop/data/dataset-Cityscapes-HDF5/'; % Luca mac
        % mrfFolderPath = '/home/siyi/dataset-Cityscapes-HDF5/'; % Siyi
        mrfFolderName = horzcat(datasetName,'-1000SP/');
        % mrfFolderName = horzcat(datasetName,'-1000SP-5classes/');
        % warning('using 5 class dataset')
    end
else
    datasetName = 'train';
    datasetPath = '/mnt/dataset/DAVIS';  % Siyi
    %     datasetPath = '/home/luca/Desktop/DAVIS'; % Luca
end

%% Constants
if useCityscape
    frontendAppName = 'frontend-bonnet-hdf5'; % Cityscapes dataset based on bonnet
else
    frontendAppName = 'frontend-DAVIS-hdf5'; % DAVIS dataset based on ground truth
end
% parameter file
paramFileName = '../tests/data/fusesParameters.yaml';

% backend optimizers
fusesAppName = 'optimization-FUSES';
fusesDAAppName = 'optimization-FUSES-DA';
fuses2AppName = 'optimization-FUSES2';
fuses2DAAppName = 'optimization-FUSES2-DA';
aeAppName = 'expansion-move-alphaExpansion';
lbpAppName = 'belief-propagation-LBP';
trwsAppName = 'belief-propagation-TRWS';
% text file containing the list of MRF files
tempFileName = 'tempNames.txt';

%% settings for plots
dim = 20; % fontsize
% markers for plots
fusesRelaxMarker = ':b';
fusesRoundedMarker = '-b*';
aeMarker = '--cs';
lbpMarker = '-.k^';
trwsMarker = '-ro';
fusesDARelaxMarker = ':r';
fusesDARoundedMarker = '-r*';
fuses2RelaxMarker = ':c';
fuses2RoundedMarker = '-c*';
fuses2DARelaxMarker = ':m';
fuses2DARoundedMarker = '-m*';
exactMarker = '-g';

% Commands
if useCityscape
    frontendCommand = horzcat(nrCoresStr,' ../build/bin/%s ', datasetPath, ' ',datasetName, ' ../bonnet/frozen/ ', paramFileName);
else
    frontendCommand = horzcat(nrCoresStr,' ../build/bin/%s ', datasetPath, ' ',datasetName, ' ', paramFileName);
end

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
nrFrames = 3;
if(nrFrames ~= length(textFiles{1})/2) warning('nrFrames > size of datasets: not running all frames'); end
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
    
    % alpha-expansion
    aeCommand = sprintf(backendCommand, aeAppName, hdfFileName, labelGTFileName);
    disp(' ')
    disp(aeCommand)
    system(aeCommand);
    
    % LBP
    lbpCommand = sprintf(backendCommand, lbpAppName, hdfFileName, labelGTFileName);
    disp(' ')
    disp(lbpCommand)
    system(lbpCommand);
    
    % TRW-S
    trwsCommand = sprintf(backendCommand, trwsAppName, hdfFileName, labelGTFileName);
    disp(' ')
    disp(trwsCommand)
    system(trwsCommand);
    
    % fusesDA
    fusesDACommand = sprintf(backendCommand, fusesDAAppName, hdfFileName, labelGTFileName);
    disp(' ')
    disp(fusesDACommand)
    system(fusesDACommand);
    fprintf('')
    
    % fuses2
    fuses2Command = sprintf(backendCommand, fuses2AppName, hdfFileName, labelGTFileName);
    disp(' ')
    disp(fuses2Command)
    system(fuses2Command);
    fprintf('')
    
    % fuses2DA
    fuses2DACommand = sprintf(backendCommand, fuses2DAAppName, hdfFileName, labelGTFileName);
    disp(' ')
    disp(fuses2DACommand)
    system(fuses2DACommand);
    fprintf('')
    
    % exact solution using CPLEX
    if useHDF5
        [label, fval_cplex] = computeExact(horzcat(mrfFolderPath,mrfFolderName,hdfFileName));
        fval_exact(i) = fval_cplex;
        labels_exact{i} = label;
    else
        [label, fval_cplex] = computeExact(hdfFileName);
        fval_exact(i) = fval_cplex;
        labels_exact{i} = label;
    end
end

%% Parsing results
nrNodes = zeros(1, nrFrames); % counts nr of nodes for each dataset
correctLabelsPerc = zeros(nrTechniques, nrFrames); % proportion against GT labels
nrSuboptLabels = zeros(nrTechniques, nrFrames);  % number against CPLEX labels
time = zeros(nrTechniques, nrFrames);
accuracyCplex = zeros(1, nrFrames);
for i = 1:nrFrames
    fprintf('Parsing results for frame %d\n',i)
    hdfFileName = hdfFiles{i};
    labelGTFileName = labelGTFiles{i};
    
    if useHDF5
        hdfFileNameNoExtension = horzcat(mrfFolderPath,mrfFolderName,hdfFileName(1:end-3));
    else
        hdfFileNameNoExtension = hdfFileName(1:end-3);
    end
    
    fusesDataFile = horzcat(hdfFileNameNoExtension,'_FUSES.csv');
    % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
    fusesData_overall = csvread(fusesDataFile, 0, 1, [0,1,4,1]);
    % store number of nodes:
    nrNodes(i) = fusesData_overall(2);
    fusesLabel = csvread(fusesDataFile, 5, 1, [5,1,5,nrNodes(i)])'; % read labels
    fusesIterationsData = csvread(fusesDataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]
    
    aeDataFile = horzcat(hdfFileNameNoExtension, '_AE.csv');
    % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels]
    aeData_overall = csvread(aeDataFile, 0, 1, [0,1,3,1]);
    aeLabel = csvread(aeDataFile, 4, 1, [4,1,4,nrNodes(i)])';
    aeIterationsData = csvread(aeDataFile, 6, 0);
    
    lbpDataFile = horzcat(hdfFileNameNoExtension, '_LBP.csv');
    lbpData_overall = csvread(lbpDataFile, 0, 1, [0,1,3,1]);
    lbpLabel = csvread(lbpDataFile, 4, 1, [4,1,4,nrNodes(i)])';
    lbpIterationsData = csvread(lbpDataFile, 6, 0);
    
    trwsDataFile = horzcat(hdfFileNameNoExtension, '_TRWS.csv');
    trwsData_overall = csvread(trwsDataFile, 0, 1, [0,1,3,1]);
    trwsLabel = csvread(trwsDataFile, 4, 1, [4,1,4,nrNodes(i)])';
    trwsIterationsData = csvread(trwsDataFile, 6, 0);
    
    fusesDADataFile = horzcat(hdfFileNameNoExtension,'_FUSESDA.csv');
    % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
    fusesDAData_overall = csvread(fusesDADataFile, 0, 1, [0,1,4,1]);
    fusesDALabel = csvread(fusesDADataFile, 5, 1, [5,1,5,nrNodes(i)])'; % read labels
    fusesDAIterationsData = csvread(fusesDADataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]
 
    fuses2DataFile = horzcat(hdfFileNameNoExtension,'_FUSES2.csv');
    % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
    fuses2Data_overall = csvread(fuses2DataFile, 0, 1, [0,1,4,1]);
    fuses2Label = csvread(fuses2DataFile, 5, 1, [5,1,5,nrNodes(i)])'; % read labels
    fuses2IterationsData = csvread(fuses2DataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]
    
    fuses2DADataFile = horzcat(hdfFileNameNoExtension,'_FUSES2DA.csv');
    % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
    fuses2DAData_overall = csvread(fuses2DADataFile, 0, 1, [0,1,4,1]);
    fuses2DALabel = csvread(fuses2DADataFile, 5, 1, [5,1,5,nrNodes(i)])'; % read labels
    fuses2DAIterationsData = csvread(fuses2DADataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]
    
    % compute accuracy for cplex solution against ground-truth
    if useHDF5
        labelGT = csvread(horzcat(mrfFolderPath, mrfFolderName, '/', labelGTFileName));
    else
        labelGT = csvread(labelGTFileName);
    end
    accuracyCplex(1, i) = (1-nnz(labels_exact{i} - labelGT)/nrNodes(i)) * 100;
    
    % put summary statistics in matrices
    % log correct labels for each method
    correctLabelsPerc(1, i) = fusesData_overall(4)/nrNodes(i) * 100;
    correctLabelsPerc(2, i) = aeData_overall(4)/nrNodes(i) * 100;
    correctLabelsPerc(3, i) = lbpData_overall(4)/nrNodes(i) * 100;
    correctLabelsPerc(4, i) = trwsData_overall(4)/nrNodes(i) * 100;
    correctLabelsPerc(5, i) = fusesDAData_overall(4)/nrNodes(i) * 100;
    correctLabelsPerc(6, i) = fuses2Data_overall(4)/nrNodes(i) * 100;
    correctLabelsPerc(7, i) = fuses2DAData_overall(4)/nrNodes(i) * 100;
    
    % log miss labels against cplex
    nrSuboptLabels(1, i) = nnz(labels_exact{i} - fusesLabel);
    nrSuboptLabels(2, i) = nnz(labels_exact{i} - aeLabel);
    nrSuboptLabels(3, i) = nnz(labels_exact{i} - lbpLabel);
    nrSuboptLabels(4, i) = nnz(labels_exact{i} - trwsLabel);
    nrSuboptLabels(5, i) = nnz(labels_exact{i} - fusesDALabel);
    nrSuboptLabels(6, i) = nnz(labels_exact{i} - fuses2Label);
    nrSuboptLabels(7, i) = nnz(labels_exact{i} - fuses2DALabel);
    
    % log time
    time(1, i) = fusesIterationsData(end, 3);
    time(2, i) = aeIterationsData(end, 3);
    time(3, i) = lbpIterationsData(end, 3);
    time(4, i) = trwsIterationsData(end, 3);
    time(5, i) = fusesDAIterationsData(end, 3);
    time(6, i) = fuses2IterationsData(end, 3);
    time(7, i) = fuses2DAIterationsData(end, 3);
    
    % plot results
    if i <= 10 % plot only first few figures
        figure; hold on
        plot(fusesIterationsData(:, 3), fusesIterationsData(:, 2),fusesRelaxMarker,'linewidth',2);
        plot(fusesIterationsData(:, 3), fusesIterationsData(:, 4),fusesRoundedMarker,'linewidth',2);
        plot(aeIterationsData(:, 3), aeIterationsData(:, 2),aeMarker,'linewidth',2);
        plot(lbpIterationsData(:, 3), lbpIterationsData(:, 2),lbpMarker,'linewidth',2);
        plot(trwsIterationsData(:, 3), trwsIterationsData(:, 2),trwsMarker,'linewidth',2);
%         plot(fusesDAIterationsData(:, 3), fusesDAIterationsData(:, 2),fusesDARelaxMarker,'linewidth',2);
%         plot(fusesDAIterationsData(:, 3), fusesDAIterationsData(:, 4),fusesDARoundedMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fusesDAIterationsData(end, 2)*ones(1, 2),fusesDARelaxMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fusesDAIterationsData(end, 4)*ones(1, 2),fusesDARoundedMarker,'linewidth',2);
%         plot(fuses2IterationsData(:, 3), fuses2IterationsData(:, 2),fuses2RelaxMarker,'linewidth',2);
%         plot(fuses2IterationsData(:, 3), fuses2IterationsData(:, 4),fuses2RoundedMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fuses2IterationsData(end, 2)*ones(1, 2),fuses2RelaxMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fuses2IterationsData(end, 4)*ones(1, 2),fuses2RoundedMarker,'linewidth',2);
%         plot(fuses2DAIterationsData(:, 3), fuses2DAIterationsData(:, 2),fuses2DARelaxMarker,'linewidth',2);
%         plot(fuses2DAIterationsData(:, 3), fuses2DAIterationsData(:, 4),fuses2DARoundedMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fuses2DAIterationsData(end, 2)*ones(1, 2),fuses2DARelaxMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fuses2DAIterationsData(end, 4)*ones(1, 2),fuses2DARoundedMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fval_exact(i)*ones(1, 2),exactMarker,'linewidth',2);
        plot([0 max(time(1:4,i))], fval_exact(i)*ones(1, 2),exactMarker,'linewidth',2);
        legend('Fuses-relaxed', 'Fuses-rounded', 'Alpha-Expansion', 'Loopy Belief Propagation', ...
            'TRW-S', 'FusesDA-relaxed','FusesDA-rounded', ...
            'Fuses2-relaxed','Fuses2-rounded', ...
            'Fuses2DA-relaxed','Fuses2DA-rounded', ...
            'Exact')
%         for t = 1:length(fuses2DAIterationsData(:, 3))-1
%             if (fuses2DAIterationsData(t, 3) - fuses2DAIterationsData(t+1, 3) == 0)
%                 plot(fuses2DAIterationsData(t, 3), fuses2DAIterationsData(t, 2),'o','markersize',10);
%                 plot(fuses2DAIterationsData(t, 3), fuses2DAIterationsData(t+1, 2),'o','markersize',10);
%             end
%         end
        xlabel('Time (ms)')
        ylabel('Function values')
        
        if useCityscape
            titleStr = horzcat(datasetName, '\_', hdfFileName(end-15:end-10), ...
                '\_', hdfFileName(end-8:end-3));
        else
            titleStr = horzcat(datasetName, '\_', hdfFileName(1:end-3));
        end
        title(titleStr)
        grid on;
        set(gca,'FontSize',dim);
        ylabh=get(gca,'ylabel'); set(ylabh, 'FontSize', dim);
        xlabh=get(gca,'ylabel'); set(xlabh, 'FontSize', dim);
        if saveEps
            %filename = horzcat(mainFolder,'TODO');
            %saveas(f,filename,'epsc');
        end
        hold off
    end
    disp_Fuses2DA_Fuses
end

%% Compute statistics
disp('Mean accuracy with standard deviation')
disp(horzcat(mean(correctLabelsPerc, 2), std(correctLabelsPerc, 0, 2)))

disp('Mean nr suboptimal labels with standard deviation')
disp(horzcat(mean(nrSuboptLabels, 2), std(nrSuboptLabels, 0, 2)))

disp('Mean runtime with standard deviation')
disp(horzcat(mean(time, 2), std(time, 0, 2)))

disp('Mean cplex accuracy against GT with standard deviation')
disp(horzcat(mean(accuracyCplex), std(accuracyCplex)))
