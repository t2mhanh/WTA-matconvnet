close all
clear all
% clc

GtDir = './ucsd_data/UCSDPed2/Test_gt';

savePath = './data/Ped2_5frame_27epoch_pa24_str12_magthres10/localSVM';
savePath1 = fullfile(savePath,'Grid1x1');

TestFoldersResult = [];
for numTestFolders = 1:12
    numTestFolders
    % load ground truth maps
    if numTestFolders < 10
        GroundTruthPath = fullfile(GtDir,['Test00' num2str(numTestFolders) '_gt.mat']);
    else
        GroundTruthPath = fullfile(GtDir,['Test0' num2str(numTestFolders) '_gt.mat']);
    end
    load(GroundTruthPath)
    groundTruthAll = M./max(M(:)); %M[nr x nc x numFrames]
    clear M

    load(fullfile(savePath1,['decisionMap_' num2str(numTestFolders) '.mat']));
    nSam = size(decisionMap,3);
    %%%
    Result_Thresh = [];
    for frame_num = 1:nSam
        % ground Truth for correspond frame
%         groundTruth = groundTruthAll(:,:,frame_num+2);
        groundTruth = groundTruthAll(:,:,frame_num+3); %+1 for Nfr=1, +3 for Nfr = 5
        [nr,nc] = size(groundTruth);
        groundTruth_ = sum(groundTruth(:));
        cur_decisionMap = decisionMap(:,:,frame_num);
        cur_decisionMap = imresize(cur_decisionMap,size(groundTruth),'bilinear');
        Result_ = [];

  	for thres = -10:0.1:20

            anomaly_map = zeros(nr,nc);
            %
           [I,J] = find(cur_decisionMap >= thres);
            for i = 1 : length(I)
                anomaly_map(I(i),J(i)) = 1;
            end
% % % % % % % %%%%%%%%%%%%%%%%%%%% EVALUATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            MyMask = sum(anomaly_map(:));
            overlap = and(groundTruth,anomaly_map);
            if sum(groundTruth(:))~= 0
                    overlap_score = sum(overlap(:))/sum(groundTruth(:))*100;
            else
                    overlap_score = 0;
            end
            NegFrame = (groundTruth_ == 0);
            PosFrame = 1 - NegFrame;
            FPFrame = ((MyMask~=0)&(groundTruth_==0));

            % # of true-positive frame at frame level
            TPFrame = ((MyMask~=0)&(groundTruth_~=0));

%             % checked evaluation (25Jan)
%             FNFrame = ((MyMask==0)&(groundTruth_~=0));
%             TNFrame = ((MyMask==0)&(groundTruth_==0));

            % # of true-positive frame at pixel level
            TPFrame_pixel = (overlap_score >= 40);
            Result_ = [Result_ [thres; TPFrame; TPFrame_pixel; PosFrame; FPFrame; NegFrame]];
%
        end
        Result_Thresh(:,:,frame_num) = Result_;
    end

    Results = sum(Result_Thresh(2:end,:,:),3);
    Thresh = Result_Thresh(1,:,frame_num);
    TestFoldersResult(:,:,numTestFolders) = [Thresh;Results];
%
end
% save(fullfile(savePath1,'Pixel_TestFoldersResults.mat'),'TestFoldersResult')
%% DRAW ROC
% pixel level
PixelLevel = zeros(3,size(Thresh,2));
TestFoldersResult_ = sum(TestFoldersResult(2:end,:,:),3);
PixelLevel(1,:) = Thresh;
PixelLevel(2,:) = TestFoldersResult_(2,:)./TestFoldersResult_(3,:); % Results = [ TPFrame; TPFrame_pixel; PosFrame; FPFrame; NegFrame]
PixelLevel(3,:) = TestFoldersResult_(4,:)./TestFoldersResult_(5,:);
PixelLevel(:,end+1) = [PixelLevel(1,end)+0.1 0 0];
%
% Area Under Curve
PixelAUC = trapz(PixelLevel(3,:),PixelLevel(2,:));
sprintf('AUC for Pixel level ROC: %d', PixelAUC)
[x_pixel,y_pixel] = EER(PixelLevel(3,:),PixelLevel(2,:));

sprintf('patial pixel EER: %d',x_pixel)
% save(fullfile(savePath1,'EER_AUC.mat'),'PixelAUC','x_pixel')

%% plot the ROC
figure(1)
plot(PixelLevel(3,:),PixelLevel(2,:),'y*-')
title('ROC curve on Ped1 dataset')
xlabel('FPR')
ylabel('TPR')
legend('Pixel level','Location','NorthWest')
hold on
x = 0:0.1:1;
y = 1 - x;
plot(y,x,'r:')
print('-f1','Ped2_pixel_level','-dpng')
% quit;
