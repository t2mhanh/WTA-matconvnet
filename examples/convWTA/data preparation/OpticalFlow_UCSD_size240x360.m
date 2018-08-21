% Input: M video sequences [nrxncxnframes]
% Output: Flows [nr x nc x nch] nch = 2 flow vx and vy

clear all

addpath('mex');
Data = 'UCSDped1';% UCSDped1/UCSDped2
numSeqs = 34; % UCSDPed1: 34, UCSDPed2: 16
DataPath = fullfile('../ucsd_data',Data,'Train');
SaveDir = fullfile('../ucsd_data','FlowResize240x360',Data,'Train');
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

ImSize = [240 360];
id = 1;
for folder_id = numSeqs
    folder_id
    if folder_id <10
        load(fullfile(DataPath,[ 'Train00' num2str(folder_id) '.mat']));%M
    else
        load(fullfile(DataPath,['Train0' num2str(folder_id) '.mat']));%M
    end
    t = size(M,3);
    Flow = [];
    max_value = [];
    min_value = [];

    im1 = M(:,:,1);
    im1 = im1/255;
    im1 = imresize(im1,ImSize,'nearest');

    for frame_num = 1:t-1
        im2 = M(:,:,frame_num+1);
        im2 = im2/255;
        im2 = imresize(im2,ImSize,'nearest');
        [vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);

        Flow(:,:,1) = vx;
        Flow(:,:,2) = vy;
        save(fullfile(SaveDir,['Flow'  num2str(id) '.mat']),'Flow')
        id = id + 1;
        im1 = im2;
    end

end
