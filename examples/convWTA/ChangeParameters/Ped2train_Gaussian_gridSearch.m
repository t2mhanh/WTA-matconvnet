% Gaussian normalization and Gaussian kernel
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

% Gaussian standardization
dataPath = './data_1/Ped2_5frame_27epoch_pa24_str12_GaussianStd';

%-------------------------------------------------------------------------------------------------------------
%%%%%%%%%%% train global SVM Grid 1x1

 load(fullfile(dataPath,'GlobalData_Grid1x1.mat'))%dataGrid, minimums, ranges
%standardization
 GlobalData = bsxfun(@minus,GlobalData,global_mean);
 GlobalData = bsxfun(@rdivide,GlobalData,global_std);
 GlobalData = double(GlobalData);
 labels = ones(size(GlobalData,1),1);
grid_matrix = zeros(325,5);
id = 1;
for i = -12:1:0
 for j = -12:1:12
 id
 disp('Train SVM')
 nu = 2^i;
 gamma = 2^j;
 param = ['-q -s 2 -n ', num2str(nu), ' -g ', num2str(gamma)];
 check = 0;

% start_time = tic;
 model = svmtrain(labels, GlobalData, param);
 [predicted_labels] = svmpredict(labels, GlobalData, model);
 inside_indices = predicted_labels > 0;
 grid_matrix(id,1) =  i;
 grid_matrix(id,2) = j;
 grid_matrix(id,3) = sum(inside_indices)/size(inside_indices,1);
 grid_matrix(id,4) = model.totalSV;
 grid_matrix(id,5) = model.rho;
 id = id + 1;
 end
end
% train_time = toc(start_time);
% disp(['Training time: ', train_time, 's'] )
save(fullfile(dataPath,'grid_matrix.mat'),'grid_matrix')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%
