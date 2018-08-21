% Gaussian normalization + linear SVM kernel
% Run Ped2train_linearSVMkernel.m before runing this file
close all
clear all
clc
run('../../matlab/vl_setupnn.m')
addpath('~/libsvm-3.22/matlab')

stride = 12;
mag_thres = 10;
patch_size = 24; % pooling_size = patch_size - 12;
pool_size = patch_size - 12;

% %% prepare data for each region
opts.modelPath = './model/convWTA_xavierImproved_pa48/net-epoch-27.mat';


%%%
DataDir = './ucsd_data/FlowResize156x240/UCSDPed2/Train'

% linear SVM kernel
dataPath = './data_1/Ped2_5frame_27epoch_pa24_str12_linearSVMkernel';
savePath = './data_1/Ped2_5frame_27epoch_pa24_str12_linearSVMkernel_GaussianNorm';
if ~exist(savePath,'dir'); mkdir(savePath); else end

%%%%%%%%%%% train global SVM Grid 1x1

 load(fullfile(dataPath,'GlobalData_Grid1x1.mat'))%dataGrid, minimums, ranges
 global_mean = mean(GlobalData,1);
 global_std  = std(GlobalData,0,1);

 %standardization
 GlobalData = bsxfun(@minus,GlobalData,global_mean);
 GlobalData = bsxfun(@rdivide,GlobalData,global_std);
 disp('Train SVM')
 param = '-q -s 2 -t 0';
 GlobalData = double(GlobalData);
 labels = ones(size(GlobalData,1),1);
 start_time = tic;
 model = svmtrain(labels, GlobalData, param);
 train_time = toc(start_time)
 %disp(['Training time:',train_time,'s'] )


 save(fullfile(savePath,'localSVMGrid1x1_GlobalMeanStd.mat'),'model','global_mean','global_std')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%
