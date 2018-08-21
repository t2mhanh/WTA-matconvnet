function Op_convWTA_patchData48_train(varargin)

run('../../matlab/vl_setupnn.m') ;

% data paths
% a path for saved models
opts.expDir = './model/convWTA_xavierImproved_pa48';
if ~exist(opts.expDir,'dir');mkdir(opts.expDir) ; else end

opts.dataDir = {'./ucsd_data/OptPatches_48'};
opts.dataType = '*.mat';

[opts, varargin] = vl_argparse(opts, varargin) ;
% experiment setup
imdbPath = './ucsd_data';
opts.imdbPath = fullfile(imdbPath,'UCSD_Optpatch48_imbd.mat');

% initialization
% opts.weightInitMethod = 'xavier';
opts.weightInitMethod = 'xavierimproved';
% opts.weightInitMethod = 'gaussian';

% training options
opts.train.batchSize = 100 ;
opts.train.numSubBatches = 1;
opts.train.continue = true;
opts.train.gpus = 1 ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 1e-4;
opts.train.numEpochs = 27
% opts.train.solverType = 'Adam';
opts.train.derOutputs = {'error', 1} ;
opts = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = UCSDTrainDataSetup('dataDir', opts.dataDir,'dataType',opts.dataType) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get training and test/validation subsets
train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;

% %--------------------------------------------------------------------------
% % set up model
% %--------------------------------------------------------------------------
net = Op_convWta_AE_3layer_init(2,[5 11],128,2,5,'weightInitMethod',opts.weightInitMethod);% dif. fanin for conv and deconv

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------


bopts.useGpu = numel(opts.train.gpus) > 0 ;
% Launch SGD

info = H_cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

% info = H_cnn_train_dag_OptimizationOpts(net, imdb, getBatchWrapper(bopts), opts.train, ...
%   'train', train, ...
%   'val', val) ;
end
% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) H_Opt_getBatch(imdb,batch,opts,'prefetch',nargout==0) ;
end


function y = H_Opt_getBatch(imdb, images, varargin)

% GET_BATCH  Load, preprocess, and pack images for CNN evaluation
opts.prefetch = false ;
opts.useGpu = false ;
opts = vl_argparse(opts, varargin);

if opts.prefetch
  % to be implemented
  data = [] ;
  return ;
end
%%
for i = 1:numel(images)
        dataPath = sprintf(fullfile(imdb.images.path{images(i)}, imdb.images.name{images(i)}));
        load(dataPath);
        data(:,:,:,i) = single(Flow_48);
end
    if opts.useGpu
      data = gpuArray(data) ;
    end
    y = {'input', data, 'target', data} ;

end
