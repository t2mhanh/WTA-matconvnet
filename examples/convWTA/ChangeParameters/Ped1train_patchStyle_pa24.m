
close all
clear all
clc
run('../../matlab/vl_setupnn.m')
addpath('~/libsvm-3.22/matlab')


stride = 12;
mag_thres = 50;
patch_size = 24;
pool_size = patch_size;
dim = 128;

% %% prepare data for each region
opts.modelPath = './model/convWTA_xavierImproved_pa48/net-epoch-27.mat';


%%%
DataDir = './ucsd_data/FlowResize156x240/UCSDPed1/Train'
savePath = ['./data/Ped1_5frame_27epoch_patchStyle_pa' num2str(patch_size) '_str' num2str(stride) '_magthres' num2str(mag_thres) '_dim' num2str(dim) '/localSVM'];
if ~exist(savePath,'dir'); mkdir(savePath); else end


%-------------------------------------------------------------------------------------------------------------
 net = load(opts.modelPath) ;
 net = dagnn.DagNN.loadobj(net.net) ;
 net.removeLayer('convt1')
 net.removeLayer('spatialsparsity1')
%net.layers(1).block.pad = [0 0 0 0];
% net.layers(3).block.pad = [0 0 0 0];
% net.layers(5).block.pad = [0 0 0 0];
 net.addLayer('pooling', ...
      dagnn.Pooling('method','max', 'poolSize', [pool_size pool_size],'pad',[0 0 0 0],'stride',stride), ...
      'x6', 'x7');

 net.mode = 'test' ;
 inputVar = 'input' ;
 predVar = net.getVarIndex('x7') ;

 useGpu = 1;
 if useGpu
 gpuDevice(useGpu)
 net.move('gpu') ;
 end
%
 % identify size of higher level grid
 load(fullfile(DataDir,'Flow156x240_1.mat'))
 hi = conv2(Flow(:,:,1,1),ones(patch_size,patch_size),'valid');
 local_ = hi(1:stride:end,1:stride:end,1,1);
 %

 for trainseq = 1:34
     trainseq
    OptPath = fullfile(DataDir,['Flow156x240_' num2str(trainseq) '.mat']);

    load(OptPath)%Flow
    nfr = size(Flow,4);
    dataGrid_local = cell(size(local_));
    GlobalData_local = [];
    for fr = 1:nfr-4
        curfl = Flow(:,:,:,fr:fr+4);
        mag = sum(curfl(:,:,:,3).^2,3);
        mag_ = conv2(mag,ones(patch_size,patch_size),'valid');
        foregr_mag = mag_(1:stride:end,1:stride:end) > mag_thres;

        for r = 1:size(foregr_mag,1)
            for c = 1:size(foregr_mag,2)
                if foregr_mag(r,c) == 1
	            curpa = curfl(stride*(r-1)+1:stride*(r-1)+patch_size,stride*(c-1)+1:stride*(c-1)+patch_size,:,:);
                    net.eval({inputVar,single(gpuArray(curpa))});
                    curdata = reshape(mean(gather(net.vars(predVar).value),4),1,dim);
                    if length(dataGrid_local{r,c}) == 0
                        dataGrid_local{r,c} = curdata;
                    else
                        dataGrid_local{r,c} = [dataGrid_local{r,c};curdata];
                    end
                    if size(GlobalData_local,1) == 0;
			GlobalData_local = curdata;
                    else
			GlobalData_local = [GlobalData_local; curdata];
		    end
                end
            end
	end
    end
    save(fullfile(savePath,['data' num2str(trainseq) '.mat']),'dataGrid_local', 'GlobalData_local')
 end

 GlobalData = [];
 dataGrid = cell(size(local_));
 for trainseq = 1:34
     trainseq
    load(fullfile(savePath,['data' num2str(trainseq) '.mat']))%GlobalData_local

    if size(GlobalData,1) == 0; GlobalData = GlobalData_local;
    else GlobalData = [GlobalData; GlobalData_local];
    end
    for r = 1:size(dataGrid,1)
        for c = 1:size(dataGrid,2)
            if length(dataGrid_local{r,c}) ~= 0
                if length(dataGrid{r,c}) == 0
                    dataGrid{r,c} = dataGrid_local{r,c};
                else
                    dataGrid{r,c} = [dataGrid{r,c};dataGrid_local{r,c}];
                end
            end
        end
    end
 end
 size(dataGrid{12,1})
 minimums = min(GlobalData_local,[],1);
 ranges = max(GlobalData_local,[],1) - minimums;
 save(fullfile(savePath,'LocalData_Grid12x18.mat'),'dataGrid','minimums','ranges','-v7.3')
 save(fullfile(savePath,'GlobalData_Grid1x1.mat'),'GlobalData','minimums','ranges','-v7.3')
 %% train local SVM
% load(fullfile(savePath,'LocalData_Grid12x18.mat'))%dataGrid, minimums, ranges
% disp('Train SVM')
% % localSVM = cell(12,18);
% localSVM = cell(size(dataGrid));
% nu =  2^-9;
% gamma = 2^-7;
% param = ['-q -s 2 -n ', num2str(nu), ' -g ', num2str(gamma)];
% check = 0;
% for i = 1:size(dataGrid,1)
%     for j = 1:size(dataGrid,2)
%         check
%         data = dataGrid{i,j};
%         if size(data,1) ~= 0
%            % SVM training
%             data = (data - repmat(minimums, size(data, 1), 1)) ./ repmat(ranges, size(data, 1), 1);
%             data = double(data);
%             labels = ones(size(data,1),1);
%             model = svmtrain(labels, data, param);
%             localSVM{i,j} = model;
%         end
%         % monitoring the training process
%         check = check + 1;
%     end
% end
% save(fullfile(savePath,'localSVM_GlobalMaxMin.mat'),'localSVM','minimums','ranges')

%%%%%%%%%%% train global SVM Grid 1x1

% load(fullfile(savePath,'GlobalData_Grid1x1.mat'))%dataGrid, minimums, ranges
% disp('Train SVM')
% nu = 2^-9;
% gamma = 2^-7;
% param = ['-q -s 2 -n ', num2str(nu), ' -g ', num2str(gamma)];
% check = 0;
% GlobalData = (GlobalData - repmat(minimums, size(GlobalData, 1), 1)) ./ repmat(ranges, size(GlobalData, 1), 1);
% GlobalData = double(GlobalData);
% labels = ones(size(GlobalData,1),1);
% model = svmtrain(labels, GlobalData, param);
% save(fullfile(savePath,'localSVMGrid1x1_GlobalMaxMin.mat'),'model','minimums','ranges')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% train SVM 6x9 %%%%%%%%%%%%%%%%%%%%%%%%%

load(fullfile(savePath,'LocalData_Grid12x18.mat'))%dataGrid, minimums, ranges
data = dataGrid;
clear dataGrid
dataGrid = cell(6,9);
for i = 1:size(data,1)
    for j = 1:size(data,2)
        r = min(ceil(i/2),size(dataGrid,1));
        c = min(ceil(j/2),size(dataGrid,2));
        if length(dataGrid{r,c}) == 0
           dataGrid{r,c} = data{i,j};
        else
           dataGrid{r,c} = [dataGrid{r,c};data{i,j}];
        end
    end
end
save(fullfile(savePath,'LocalData_Grid6x9.mat'),'dataGrid','minimums','ranges','-v7.3')
%% train local SVM
% load(fullfile(savePath,'LocalData_Grid6x9.mat'))%dataGrid, minimums, ranges
disp('Train SVM')
localSVM = cell(6,9);
nu = 2^-9;
gamma = 2^-7;
param = ['-q -s 2 -n ', num2str(nu), ' -g ', num2str(gamma)];
check = 0;
for i = 1:size(dataGrid,1)
    for j = 1:size(dataGrid,2)
        check
        data = dataGrid{i,j};
        if size(data,1) ~= 0
           % SVM training
            data = (data - repmat(minimums, size(data, 1), 1)) ./ repmat(ranges, size(data, 1), 1);
            data = double(data);
            labels = ones(size(data,1),1);
            model = svmtrain(labels, data, param);
            localSVM{i,j} = model;
        end
        % monitoring the training process
        check = check + 1;
    end
end
save(fullfile(savePath,'localSVMGrid6x9_GlobalMaxMin.mat'),'localSVM','minimums','ranges')
