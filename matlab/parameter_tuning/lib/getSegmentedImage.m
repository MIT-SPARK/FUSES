function imageSeg = getSegmentedImage(x, K, N, imageRGB, imageLabels, spIndices)
% function generating image matrix from segmentation result
% Date: 2/2/2018
% Authors: Luca Carlone, Siyi Hu

% create a cell array of K empty images
imageEmpty = zeros(size(imageRGB), 'uint8');
imageSeg = cell(1, K);
for k = 1:K
    imageSeg{k} = imageEmpty;
end

% add pixels to the corresponding segmented plot
for i = 1:N
    k = (1:K)*x((i-1)*K+1:i*K);              % class of superpixel i
    
    % if super pixel i is not in any class or in more than one class
    if k < 1 || k > K                
        k = 1;      % set it to background
        fprintf('super pixel %i not labeled\n', i);
    end
    
    pixels = spIndices{i};                   % pixels in superpixel i
    for j = 1:length(pixels)
        p = pixels(j);
        [r, c] = ind2sub(size(imageLabels), p);
        imageSeg{k}(r, c, :) = imageRGB(r, c, :);% add pixel to corresponding result image
    end
end

% % segmented image
% for k = 1:K
%     figure
%     imshow(imageSeg{k})
%     title(['Region ' num2str(k)])
% end
end