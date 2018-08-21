close all
clear all
clc

run('../../matlab/vl_setupnn.m')
addpath('~/libsvm-3.22/matlab')

OptDir = './ucsd_data/FlowResize156x240/UCSDPed2/Test';

stride = 12;
mag_thres = 50;
mag_thres_train = 10;
patch_size = 24;
pool_size = patch_size;
dim = 128;
Nfr = 9

%%
opts.modelPath = './model/convWTA_xavierImproved_pa48/net-epoch-27.mat';

savePath = ['./data/Ped2_5frame_27epoch_patchStyle_pa' num2str(patch_size) '_str' num2str(stride) '_magthres' num2str(mag_thres_train) '_dim' num2str(dim) '_Nfr' num2str(Nfr) '/localSVM'];
savePath1 = fullfile(savePath,'Grid1x1');
if ~exist(savePath1,'dir'); mkdir(savePath1); end

%% SVM model
load(fullfile(savePath,'localSVMGrid1x1_GlobalMaxMin.mat')) % 'localSVM' minimums ranges

%----------------------------------------------------------------------------------------------

net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;
net.removeLayer('convt1')
net.removeLayer('spatialsparsity1')
%net.layers(1).block.pad = [0 0 0 0];
%net.layers(3).block.pad = [0 0 0 0];
%net.layers(5).block.pad = [0 0 0 0];
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

min_va = 0; %
max_va = 0;%
for numTestFolders = 1:12
    numTestFolders

    OptPath = fullfile(OptDir,['Flow156x240_' num2str(numTestFolders) '.mat']);
    load(OptPath)
    nfr = size(Flow,4);
    %% CALCULATE AND SAVE ERROR MAP
    decisionMap = [];
    for fr_n = 1:nfr-Nfr + 1
        curflow = Flow(:,:,:,fr_n:fr_n+Nfr-1);
      	if Nfr > 1
            mag = sum(curflow(:,:,:,ceil(Nfr/2)).^2,3);
        else
            mag = sum(curflow.^2,3);
        end

        mag_ = conv2(mag,ones(patch_size,patch_size),'valid');
        foregr_mag = mag_(1:stride:end,1:stride:end) > mag_thres;

        decision_values = -10*ones(size(foregr_mag));
        for r = 1:size(foregr_mag,1)
            for c = 1:size(foregr_mag,2)
                if foregr_mag(r,c) == 1
            		    if Nfr > 1
                        curpa = curflow(stride*(r-1)+1:stride*(r-1)+patch_size,stride*(c-1)+1:stride*(c-1)+patch_size,:,:);
%                        size(curpa)
                        net.eval({inputVar,single(gpuArray(curpa))});
                        curdata = reshape(mean(gather(net.vars(predVar).value),4),1,dim);
                    else
                        curpa = curflow(stride*(r-1)+1:stride*(r-1)+patch_size,stride*(c-1)+1:stride*(c-1)+patch_size,:);
 %                       size(curpa)
                        net.eval({inputVar,single(gpuArray(curpa))});
                        curdata = reshape(gather(net.vars(predVar).value),1,dim);
                    end


%                    curpa = curflow(stride*(r-1)+1:stride*(r-1)+patch_size,stride*(c-1)+1:stride*(c-1)+patch_size,:,:);
%                    net.eval({inputVar,single(gpuArray(curpa))});
%                    curdata = reshape(mean(gather(net.vars(predVar).value),4),1,dim);
                    curdata_ = (curdata - minimums) ./ ranges;
                    % use svmmodel for test
                    [predicted_labels, ~ , prob_estimates] = svmpredict(1, double(curdata_),model,'-q');
                    decision_values(r,c) = - prob_estimates;
                    if -prob_estimates < min_va; min_va = - prob_estimates; end
                    if -prob_estimates > max_va; max_va = - prob_estimates; end
                end
            end
        end
        decisionMap(:,:,fr_n) = decision_values;%
    end
        save(fullfile(savePath1,['decisionMap_' num2str(numTestFolders) '.mat']),'decisionMap')

end
