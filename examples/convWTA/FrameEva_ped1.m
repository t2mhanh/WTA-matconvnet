close all
clear all

savePath = './data/Ped1_5frame_27epoch_pa24_str12_magthres10/localSVM';
savePath1 = fullfile(savePath,'Grid6x9');

%%
testSeqPath = '/ucsd_data/UCSDPed1/Test_gt';
load(fullfile(testSeqPath,'TestFrameGT.mat'))%,'FrameGt')   % FROM UCSD

TestFoldersResult = [];
for numTestFolders = 1:36
    frameGt = FrameGt{1,numTestFolders};
    numTestFolders
    load(fullfile(savePath1,['decisionMap_' num2str(numTestFolders) '.mat']));     % foregr_mask = 50
    nSam = size(decisionMap,3);
    cur_gt = frameGt(4:nSam+3);

    Result = [];
     for thres = -1:0.01:10
        MyMask = zeros(1,nSam);
        for frame_num = 1:nSam
            cur_decisionMap = decisionMap(:,:,frame_num);
%            cur_decisionMap = imresize(cur_decisionMap,[156 240],'bilinear');
            [I,J] = find(cur_decisionMap >= thres);
            if length(I) >0, MyMask(frame_num) = 1;else end
        end
        cur_mask = MyMask(1:nSam);
        PosFrame = sum(cur_gt);
        NegFrame = sum(not(cur_gt));
        TPFr = sum(and(cur_gt,cur_mask));
        FPFr = sum(and(not(cur_gt),cur_mask));
        Result = [Result [thres; TPFr;PosFrame;FPFr;NegFrame]];
    end
    TestFoldersResult(:,:,numTestFolders) = Result;
end

%% DRAW ROC curve
TestFoldersResult_ = sum(TestFoldersResult(2:end,:,:),3);
% frame level
Thresh = TestFoldersResult(1,:,numTestFolders);
FrameLevel = zeros(3,size(Thresh,2));
FrameLevel(1,:) =  Thresh;
FrameLevel(2,:) = TestFoldersResult_(1,:)./TestFoldersResult_(2,:); %TPR
FrameLevel(3,:) = TestFoldersResult_(3,:)./TestFoldersResult_(4,:); %FPR
FrameLevel(:,end+1) = [FrameLevel(1,end)+0.1 0 0];
%
% Area Under Curve
FrameAUC = trapz(FrameLevel(3,:),FrameLevel(2,:));
sprintf('AUC for Frame level ROC: %d', FrameAUC)
[x_frame,y_frame] = EER(FrameLevel(3,:),FrameLevel(2,:));
sprintf('patial pixel EER: %d',x_frame)
% save(fullfile(savePath1,'Frame_EER_AUC.mat'),'FrameAUC','x_frame')
% quit;
%% plot the ROC
figure(1)
plot(FrameLevel(3,:),FrameLevel(2,:),'gs-')
title('ROC curve on Ped1 dataset')
xlabel('FPR')
ylabel('TPR')
legend('Frame level','Location','NorthWest')
hold on
x = 0:0.1:1;
y = 1 - x;
plot(y,x,'r:')
print('-f1','Ped1_frame_level','-dpng')
