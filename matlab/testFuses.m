% Script running image segmentation for cityscapes dataset using Fuses and
% plotting function value before and after rounding at each iteration

% Date: 6/14/2018
% Authors: Luca Carlone, Siyi Hu

clear all; close all;
% Variables
datasetName = 'krefeld';

% Constants
fusesAppName = 'fuses-example-bonnet';
tempFileName = 'tempNames.txt';
datasetPath = '/mnt/dataset/Cityscapes';

% Commands
fusesCommand = horzcat('taskset -c 0 ../build/bin/%s ', datasetPath, ' ', ...
    datasetName, ' ../bonnet/frozen/');

%% Runing Fuses
disp('running script:')

% fuses
fusesCommand = sprintf(fusesCommand, fusesAppName);
disp(' ')
disp(fusesCommand)
system(fusesCommand);

% get file names from FUSES output
fileID = fopen(tempFileName);
textFiles = textscan(fileID, '%s');
nrFrames = length(textFiles{1})/2;

hdfFiles = cell(1, nrFrames);
labelGTFiles = cell(1, nrFrames);
for i = 1:nrFrames
   hdfFiles{i} = textFiles{1}{2*i-1};
   labelGTFiles{i} = textFiles{1}{2*i};
end

%% Parsing data
nrNodes = zeros(1, nrFrames);
correctLabels = zeros(1, nrFrames);
for i = 1:nrFrames
    hdfFileName = hdfFiles{i};
    labelGTFileName = labelGTFiles{i};
    
    fusesDataFile = horzcat(hdfFileName(1:end-3), '_FUSES.csv');
    fusesData_overall = csvread(fusesDataFile, 0, 1, [0,1,4,1]);
    fusesData = csvread(fusesDataFile, 7, 0);
    
    % log correct labels for each method
    nrNodes(i) = fusesData_overall(2);
    correctLabels(i) = fusesData_overall(4)/nrNodes(i);
    
    % plot results
    if i<= 5
        figure
        hold on
        plot(fusesData(:, 3), fusesData(:, 2), '.-');
        plot(fusesData(:, 3), fusesData(:, 4), '.-');
        legend('Fuses', 'Fuses after rounding')
        % xlim([0 20])
        xlabel('Time (ms)')
        ylabel('Function values')
        titleStr = horzcat(datasetName, '\_', hdfFileName(end-15:end-10), ...
            '\_', hdfFileName(end-8:end-3));
        title(titleStr)
        grid on
        hold off
    end
end
% disp('Mean accuracy')
% disp(mean(correctLabels))
% disp('Standard deviation')
% disp(std(correctLabels, 0))