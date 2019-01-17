function [imageRGBName, imageLabelsName] = CityscapesParser(testDataset, t)
% function parsing Cityscapes dataset to return image name strings for 
% frame t in the specified dataset
% Date: 6/21/2018
% Authors: Siyi Hu

% constant string
datasetFolder = '~/datasets-videoSegmentation/Cityscapes/';

% find name strings
pathToImages = horzcat(datasetFolder,'leftImg8bit_trainvaltest/leftImg8bit/train/', testDataset,'/');
pathToLabels = horzcat(datasetFolder,'gtFine_trainvaltest/gtFine/train/',testDataset,'/');

RGBlisting = dir(horzcat(pathToImages, '*.png'));
imageRGBName = horzcat(pathToImages, RGBlisting(t).name);

Labelslisting = dir(horzcat(pathToLabels, '*_labelIds.png'));
imageLabelsName = horzcat(pathToLabels, Labelslisting(t).name);
end