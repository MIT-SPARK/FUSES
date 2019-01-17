function [PA, MPA, MIoU, FWIoU] = computeAccuracyfromMatrix(P)
% function computing accuracy statistics given matrix P
% P_ij stores the amount of pixels of class i inferred to belong to class j
% Date: 7/15/2018
% Authors: Luca Carlone, Siyi Hu

% pre-processing: set everything less than 1 to 0
for i = 1:size(P, 1)
    for j = 1:size(P, 2)
        if P(i,j) < 1 || P(i,j) > 10^7
            P(i,j) = 0;
        end
    end
end
% ignore the last class in false positive computation
P(end, :) = zeros(1, size(P, 2));
           
% % remove non-present classes
% i = 1;
% while i <= size(P, 1)
%     if sum(P(i,:)) + sum(P(:,i)) > 1
%         i = i + 1;
%     else
%         P = horzcat(P(:, 1:i-1), P(:, i+1:end));
%         P = vertcat(P(1:i-1, :), P(i+1:end, :));
%     end
% end
% K = size(P, 1);

K = 0; % number of classes in ground truth
for i = 1:(size(P, 1)-1)
    if sum(P(i,:)) > 0
        K = K + 1;
    end
end

% compute accuracy
PA = sum(diag(P)) / sum(P(:));
PA = PA*100;

MPA = 0;
for i = 1:(size(P, 1)-1)
    if sum(P(i,:)) > 0 % if class i found in ground truth
        MPA = MPA + (1/K)*P(i,i)/sum(P(i,:));
    end
end
MPA = MPA*100;
    
MIoU = 0;
for i = 1:(size(P, 1)-1)
    if sum(P(i,:)) > 0
        MIoU = MIoU + (1/K)*P(i,i)/( sum(P(i,:)) + sum(P(:,i)) - P(i,i) );
    end
end
MIoU = MIoU*100;

FWIoU = 0;
for i = 1:(size(P, 1)-1)
    if sum(P(i,:)) > 0
        FWIoU = FWIoU + (1/sum(P(:)))*sum(P(i,:))*P(i,i)/...
            (sum(P(i,:)) + sum(P(:,i)) - P(i,i));
    end
end
FWIoU = FWIoU*100;
end