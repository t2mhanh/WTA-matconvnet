% Resize UCSD data to 156x240
% Use this size with One-class SVM
clear all

addpath('mex');
Data = 'UCSDped1';% UCSDped1/UCSDped2
%Type = 'Train';
Type = 'Test';%Train/Test
numSeqs = 36;
DataPath = fullfile('../ucsd_data',Data,Type);
SaveDir = fullfile('../ucsd_data/FlowResize156x240',Data,Type);
if ~exist(SaveDir,'dir'), mkdir(SaveDir),else end
% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;
para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

%max_value_folders = [];
%min_value_folders = [];

total_time = 0;
total_frame = 0;
for folder_id = 1:numSeqs
    folder_id
    tic
    if folder_id <10
        load(fullfile(DataPath,[Type '00' num2str(folder_id) '.mat']));%M
    else
        load(fullfile(DataPath,[Type '0' num2str(folder_id) '.mat']));%M
    end
    t = size(M,3);
    total_frame = total_frame + t;
    Flow = [];
    max_value = [];
    min_value = [];
    im1 = M(:,:,1);
    im1 = imresize(im1,[156,240],'nearest');
    im1 = im1/255;
    for frame_num = 1:t-1
        im2 = M(:,:,frame_num+1);
        im2 = imresize(im2,[156,240],'nearest');
        im2 = im2/255;
        [vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
        im1 = im2;
        Flow(:,:,1,frame_num) = vx;
        Flow(:,:,2,frame_num) = vy;
    end
%     save(fullfile(SaveDir,['Flow156x240_' num2str(folder_id) '.mat']),'Flow','-v7.3')
    total_time = total_time + toc;
end
