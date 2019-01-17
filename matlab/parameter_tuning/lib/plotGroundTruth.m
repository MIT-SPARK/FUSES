function plotGroundTruth(imageRGB, imageLabels, color)
% function plotting the ground truth segmentation of an image
% Date: 2/2/2018
% Authors: Luca Carlone, Siyi Hu

% check if there are enough colors
if length(color) < max(imageLabels(:))+1
    error('Not enough colors for the ground truth!') 
end

% original RGB image
figure             
subplot(1,2,1)
imshow(imageRGB)

% ground truth labels
subplot(1,2,2)
imageLabelsColor = zeros(size(imageRGB), 'uint8');
for i = 1:numel(imageLabels)
    [r, c] = ind2sub(size(imageLabels), i);
    class = imageLabels(r, c) + 1;  % add 1 so class index starts from 1
    imageLabelsColor(r, c, :) = color(class, :);
end
imshow(imageLabelsColor)

end