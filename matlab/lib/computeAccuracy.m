function [PA, MPA, MIoU, FWIoU] = computeAccuracy(labelGT_table, label)
% function computing accuracy statistics for a set of label with 
% given ground-truth label
% labelGT_table(i, j) = the number of superpixls belonging to class (j-1)
% in superpixel (i-1)
% label = a column vector of labels (class index starts from 0)
% Date: 7/14/2018
% Authors: Luca Carlone, Siyi Hu

% compare labels computed with ground truth and store P_(i+1)(j+1) (the 
% amount of pixels of class i inferred to belong to class j) in matrix P
K = size(labelGT_table, 2);
P = zeros(K, K);
for i = 1:size(labelGT_table, 1)
    classInferred = round(label(i));
%     [v, index] = max(labelGT_table(i, :));
%     classInferred = index - 1;
    P(:, classInferred+1) = P(:, classInferred+1)+labelGT_table(i, :)';
end

[PA, MPA, MIoU, FWIoU] = computeAccuracyfromMatrix(P);
end