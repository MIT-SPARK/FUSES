function spIntensity = getSPintensities(imageGray, spIndices, N)
% function computing average intensity for each superpixel
% Date: 6/15/2018
% Authors: Siyi Hu
    
spIntensity = ones(1, N); % superpixel average intensity
for i = 1:N
    spIntensity(i) = mean(imageGray(spIndices{i}));
end