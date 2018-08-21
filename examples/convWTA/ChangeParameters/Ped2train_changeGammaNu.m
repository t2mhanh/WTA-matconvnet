
close all
clear all
clc
run('../../matlab/vl_setupnn.m')
addpath('~/libsvm-3.22/matlab')

stride = 12;
mag_thres = 10;
patch_size = 24;
pool_size = patch_size - 12;

% %% prepare data for each region
opts.modelPath = './model/convWTA_xavierImproved_pa48/net-epoch-27.mat';


%%%
DataDir = './ucsd_data/FlowResize156x240/UCSDPed2/Train'
% Min-max scaling
dataPath = './data/Ped2_5frame_27epoch_pa24_str12_magthres10/localSVM';
savePath = './data_1/Ped2_5frame_27epoch_pa24_str12_gamma-12_nu-9';
if ~exist(savePath,'dir'); mkdir(savePath); else end


%%%%%%%%%%% train global SVM Grid 1x1

 load(fullfile(dataPath,'GlobalData_Grid1x1.mat'))%dataGrid, minimums, ranges
 disp('Train SVM')
 nu = 2^-9;
 gamma = 2^-12;
 param = ['-q -s 2 -n ', num2str(nu), ' -g ', num2str(gamma)];
 check = 0;
 GlobalData = (GlobalData - repmat(minimums, size(GlobalData, 1), 1)) ./ repmat(ranges, size(GlobalData, 1), 1);
 GlobalData = double(GlobalData);
 labels = ones(size(GlobalData,1),1);
 model = svmtrain(labels, GlobalData, param);
model.totalSV
 save(fullfile(savePath,'localSVMGrid1x1_GlobalMaxMin.mat'),'model','minimums','ranges')
