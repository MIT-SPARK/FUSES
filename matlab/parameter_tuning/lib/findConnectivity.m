function connMatrix = findConnectivity(L, N)
% function computing the connectivity matrix for a given set of superpixels 
% Inputs:
% - N: nr of superpixels
% - L: image where L(r,c) contains the id of the superpixel to which pixel (r,c) belong
% Date: 11/15/2017
% Authors: Luca Carlone, Siyi Hu

% Initialize connectivity matrix such that nothing is connected 
connMatrix = zeros(N, N);
for r = 1:size(L,1)-1
   for c = 1:size(L,2)-1
       % If pixel L(r,c) and the pixel to the right belong to different
       % superpixels, the two superpixels are connected
       if L(r, c) ~= L(r+1, c)
           connMatrix(L(r, c), L(r+1, c)) = 1;
           connMatrix(L(r+1, c), L(r, c)) = 1;
       end
       
       % If pixel L(r,c) and the pixel down below belongs different
       % supperpixels, the two superpixels are connected
       if L(r, c) ~= L(r, c+1)
           connMatrix(L(r, c), L(r, c+1)) = 1;
           connMatrix(L(r, c+1), L(r, c)) = 1;   
       end
   end
end
end