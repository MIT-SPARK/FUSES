% Script plotting Fuses optimality gap before rounding for various number
% of nodes and number of classes 
% Date: 7/12/2018
% Authors: Luca Carlone, Siyi Hu

% TODO: add classes scalability
clear all; close all; clc;
addpath('./lib')

%% Datasets / settings
useHDF5 = false;
datasetName = 'lindau';
datasetPath = '/mnt/dataset/Cityscapes'; % Siyi

nrNodesList = 400:200:2000;
nrClassesList = 4:4:20;

%% Constants
frontendAppName = 'frontend-bonnet-hdf5-scalability'; % Cityscapes dataset based on bonnet
paramFileName = '../tests/data/fusesParameters.yaml';

% backend optimizers
fusesAppName = 'optimization-FUSES';
fuses2DAAppName = 'optimization-FUSES2-DA';
% text file containing the list of MRF files
tempFileName = 'tempNames-scalability.txt';

%% Plot settings and Cpp commands
dim = 15; % fontsize
% colors for plot
fusesLineColor = 'b';
fuses2DALineColor = 'm';

% Commands
frontendCommand = horzcat('../build/bin/%s ', datasetPath, ' ',datasetName, ' ../bonnet/frozen/ ', paramFileName, ' %f %f');

if useHDF5
    backendCommand = horzcat(' ../build/bin/%s ', mrfFolderPath, mrfFolderName, '/%s ', mrfFolderPath, mrfFolderName, '/%s');
else
    backendCommand = horzcat(' ../build/bin/%s %s %s');
end

%% Loop through each number of nodes
% log data
fusesGap_nodes = zeros(1, length(nrNodesList));
fusesGap_std_nodes = zeros(1, length(nrNodesList));
fusesGapData_nodes = cell(1, length(nrNodesList));
fuses2DAGap_nodes = zeros(1, length(nrNodesList));
fuses2DAGap_std_nodes = zeros(1, length(nrNodesList));
fuses2DAGapData_nodes = cell(1, length(nrNodesList));
fusesNrNodes = zeros(1, length(nrNodesList));
for n = 1:length(nrNodesList)
    nrNodes = nrNodesList(n);
    fprintf('=================== nrNodes = %i ====================\n',nrNodes)
    
    %% Runing front end
    if useHDF5 % use precomputed MRF from Bonnet frontend
        disp('running script:')
    else
        disp('running script with front end:')
        command = sprintf(frontendCommand, frontendAppName, nrNodes);
        disp(' ')
        disp(command)
        system(command);
    end
    
    %% Getting file names from front end output
    fileID = fopen(tempFileName);
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
    
    %% Runing all backend algorithms
    for i = 1:nrFrames
        hdfFileName = hdfFiles{i};
        labelGTFileName = labelGTFiles{i};
        
        % fuses
        fusesCommand = sprintf(backendCommand, fusesAppName, hdfFileName, labelGTFileName);
        disp(' ')
        disp(fusesCommand)
        system(fusesCommand);
        fprintf('')
        
        % fuses2DA
        fuses2DACommand = sprintf(backendCommand, fuses2DAAppName, hdfFileName, labelGTFileName);
        disp(' ')
        disp(fuses2DACommand)
        system(fuses2DACommand);
        fprintf('')
        
        % exact solution using CPLEX
        [label, fval_cplex] = computeExact(hdfFileName);
        fval_exact(i) = fval_cplex;
        labels_exact{i} = label;
    end
    
    %% Parsing results
    nrNodes_actual = zeros(1, nrFrames);
    fusesGapData = zeros(1, nrFrames);
    fuses2DAGapData = zeros(1, nrFrames);
    for i = 1:nrFrames
        fprintf('Parsing results for frame %d\n',i)
        hdfFileName = hdfFiles{i};
        labelGTFileName = labelGTFiles{i};
        hdfFileNameNoExtension = hdfFileName(1:end-3);
        
        % read in results
        fusesDataFile = horzcat(hdfFileNameNoExtension,'_FUSES.csv');
        % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
        fusesData_overall = csvread(fusesDataFile, 0, 1, [0,1,4,1]);
        % store number of nodes:
        nrNodes_actual(i) = fusesData_overall(2);
        fusesIterationsData = csvread(fusesDataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]
        
        fuses2DADataFile = horzcat(hdfFileNameNoExtension,'_FUSES2DA.csv');
        % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
        fuses2DAData_overall = csvread(fuses2DADataFile, 0, 1, [0,1,4,1]);
        fuses2DAIterationsData = csvread(fuses2DADataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]

        % compute optimality gap
        fval_fuses = fusesIterationsData(end, 2); % fval before rounding
        fusesGapData(1, i) = (fval_fuses - fval_exact(i))/fval_exact(i) * 100;
        fval_fuses2DA = fuses2DAIterationsData(end, 2); % fval before rounding
        fuses2DAGapData(1, i) = (fval_fuses2DA - fval_exact(i))/fval_exact(i) * 100;
    end
    
    % log data for each loop
    fusesGap_nodes(1, n) = mean(fusesGapData);
    fusesGap_std_nodes(1, n) = std(fusesGapData);
    fusesGapData_nodes{1, n} = fusesGapData;
    fuses2DAGap_nodes(1, n) = mean(fuses2DAGapData);
    fuses2DAGap_std_nodes(1, n) = std(fuses2DAGapData);
    fuses2DAGapData_nodes{1, n} = fuses2DAGapData;
    fusesNrNodes(1, n) = mean(nrNodes_actual);
end

%% Loop through each number of classes
% log data
fusesGap_classes = zeros(1, length(nrClassesList));
fusesGap_std_classes = zeros(1, length(nrClassesList));
fusesGapData_classes = cell(1, length(nrClassesList));
fuses2DAGap_classes = zeros(1, length(nrClassesList));
fuses2DAGap_std_classes = zeros(1, length(nrClassesList));
fuses2DAGapData_classes = cell(1, length(nrClassesList));
fusesNrNodes2 = zeros(1, length(nrClassesList));
for c = 1:length(nrClassesList)
    nrClasses = nrClassesList(c);
    fprintf('=================== nrClasses = %i ====================\n',nrClasses)
    
    %% Runing front end
    if useHDF5 % use precomputed MRF from Bonnet frontend
        disp('running script:')
    else
        disp('running script with front end:')
        command = sprintf(frontendCommand, frontendAppName, 1000, nrClasses);
        disp(' ')
        disp(command)
        system(command);
    end
    
    %% Getting file names from front end output
    fileID = fopen(tempFileName);
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
    
    %% Runing all backend algorithms
    for i = 1:nrFrames
        hdfFileName = hdfFiles{i};
        labelGTFileName = labelGTFiles{i};
        
        % fuses
        fusesCommand = sprintf(backendCommand, fusesAppName, hdfFileName, labelGTFileName);
        disp(' ')
        disp(fusesCommand)
        system(fusesCommand);
        fprintf('')
        
        % fuses2DA
        fuses2DACommand = sprintf(backendCommand, fuses2DAAppName, hdfFileName, labelGTFileName);
        disp(' ')
        disp(fuses2DACommand)
        system(fuses2DACommand);
        fprintf('')
        
        % exact solution using CPLEX
        [label, fval_cplex] = computeExact(hdfFileName);
        fval_exact(i) = fval_cplex;
        labels_exact{i} = label;
    end
    
    %% Parsing results
    nrNodes_actual = zeros(1, nrFrames);
    fusesGapData = zeros(1, nrFrames);
    fuses2DAGapData = zeros(1, nrFrames);
    for i = 1:nrFrames
        fprintf('Parsing results for frame %d\n',i)
        hdfFileName = hdfFiles{i};
        labelGTFileName = labelGTFiles{i};
        hdfFileNameNoExtension = hdfFileName(1:end-3);
        
        % read in results
        fusesDataFile = horzcat(hdfFileNameNoExtension,'_FUSES.csv');
        % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
        fusesData_overall = csvread(fusesDataFile, 0, 1, [0,1,4,1]);
        % store number of nodes:
        nrNodes_actual(i) = fusesData_overall(2);
        fusesIterationsData = csvread(fusesDataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]
        
        fuses2DADataFile = horzcat(hdfFileNameNoExtension,'_FUSES2DA.csv');
        % read first 5 rows, containing [1: timing, 2: nrNodes, 3: nrClasses, 4: nr correct labels, 5: value after rounding]
        fuses2DAData_overall = csvread(fuses2DADataFile, 0, 1, [0,1,4,1]);
        fuses2DAIterationsData = csvread(fuses2DADataFile, 7, 0); % read last rows containing [nrIter, relaxedCost, cumTime, roundedCost]

        % compute optimality gap
        fval_fuses = fusesIterationsData(end, 2); % fval before rounding
        fusesGapData(1, i) = (fval_fuses - fval_exact(i))/fval_exact(i) * 100;
        fval_fuses2DA = fuses2DAIterationsData(end, 2); % fval before rounding
        fuses2DAGapData(1, i) = (fval_fuses2DA - fval_exact(i))/fval_exact(i) * 100;
    end
    
    % log data for each loop
    fusesGap_classes(1, c) = mean(fusesGapData);
    fusesGap_std_classes(1, c) = std(fusesGapData);
    fusesGapData_classes{1, c} = fusesGapData;
    fuses2DAGap_classes(1, c) = mean(fuses2DAGapData);
    fuses2DAGap_std_classes(1, c) = std(fuses2DAGapData);
    fuses2DAGapData_classes{1, c} = fuses2DAGapData;
    fusesNrNodes2(1, c) = mean(nrNodes_actual);
end

%% Plot results - nodes
figure
subplot(1,2,1); hold on
fuses_y1 = fusesGap_nodes + fusesGap_std_nodes; 
fuses_y2 = fusesGap_nodes - fusesGap_std_nodes; 
fuses_px=[nrNodesList,fliplr(nrNodesList)];
fuses_py=[fuses_y1, fliplr(fuses_y2)];
patch(fuses_px,fuses_py,1,'FaceColor',fusesLineColor,'EdgeColor','none');
h1 = plot(nrNodesList, fusesGap_nodes, 'color', fusesLineColor, 'linewidth',2);
fuses2DA_y1 = fuses2DAGap_nodes + fuses2DAGap_std_nodes; 
fuses2DA_y2 = fuses2DAGap_nodes - fuses2DAGap_std_nodes; 
fuses2DA_px=[nrNodesList,fliplr(nrNodesList)];
fuses2DA_py=[fuses2DA_y1, fliplr(fuses2DA_y2)];
patch(fuses2DA_px,fuses2DA_py,1,'FaceColor',fuses2DALineColor,'EdgeColor','none');
h2 = plot(nrNodesList, fuses2DAGap_nodes, 'color', fuses2DALineColor, 'linewidth',2);
xlabel('Number of Superpixels')
xlim([nrNodesList(1) nrNodesList(end)])
ylabel('Relaxation gap (%)')
ylim([0 5])
legend([h1 h2], {'FUSES', 'DARS'})
grid on;
set(gca,'FontSize',dim);
ylabh=get(gca,'ylabel'); set(ylabh, 'FontSize', dim);
xlabh=get(gca,'xlabel'); set(xlabh, 'FontSize', dim);
alpha(.2)

%% Plot results - classes
subplot(1,2,2); hold on
fuses_y1 = fusesGap_classes + fusesGap_std_classes; 
fuses_y2 = fusesGap_classes - fusesGap_std_classes; 
fuses_px=[nrClassesList,fliplr(nrClassesList)];
fuses_py=[fuses_y1, fliplr(fuses_y2)];
patch(fuses_px,fuses_py,1,'FaceColor',fusesLineColor,'EdgeColor','none');
h1 = plot(nrClassesList, fusesGap_classes, 'color', fusesLineColor, 'linewidth',2);
fuses2DA_y1 = fuses2DAGap_classes + fuses2DAGap_std_classes; 
fuses2DA_y2 = fuses2DAGap_classes - fuses2DAGap_std_classes; 
fuses2DA_px=[nrClassesList,fliplr(nrClassesList)];
fuses2DA_py=[fuses2DA_y1, fliplr(fuses2DA_y2)];
patch(fuses2DA_px,fuses2DA_py,1,'FaceColor',fuses2DALineColor,'EdgeColor','none');
h2 = plot(nrClassesList, fuses2DAGap_classes, 'color', fuses2DALineColor, 'linewidth',2);
xlabel('Number of classes')
xlim([nrClassesList(1) nrClassesList(end)])
ylabel('Relaxation gap (%)')
ylim([0 5])
legend([h1 h2], {'FUSES', 'DARS'})
grid on;
set(gca,'FontSize',dim);
ylabh=get(gca,'ylabel'); set(ylabh, 'FontSize', dim);
xlabh=get(gca,'xlabel'); set(xlabh, 'FontSize', dim);
alpha(.2)

%% save variables to .mat
filename = horzcat(datasetName, '-scalability.mat');
save(filename)