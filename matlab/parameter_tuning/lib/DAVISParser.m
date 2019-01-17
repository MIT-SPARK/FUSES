function [imageRGBName, imageLabelsName] = DAVISParser(testDataset, t)
% function parsing DAVIS dataset to return image name strings for frame t
% in the specified dataset
% Date: 6/21/2018
% Authors: Siyi Hu

% constant string
datasetFolder = '~/datasets-videoSegmentation/DAVIS/';
resolution = '480p';

pathToImages = horzcat(datasetFolder,'JPEGImages/',resolution,'/',testDataset,'/');
pathToLabels = horzcat(datasetFolder,'Annotations/',resolution,'/',testDataset,'/');

imageName = sprintf('%05d',t-1);    % image starts from 0
imageRGBName = horzcat(pathToImages,imageName,'.jpg');
imageLabelsName = horzcat(pathToLabels,imageName,'.png');

end