function [y,mask] = vl_nnSpatialSparsity(x,varargin)
% CNN Spatial Sparsity.
%   [Y,MASK] = VL_NNSPATIALSPARSITY(X) keeps THE SINGLE LARGEST value within each channel of the data X. MASK
%   is the location of the single largest value for each channel. Both Y and MASK have the
%   same size as X.
%
%
%   [DZDX] = VL_NNSPATIALSPARSITY(X, DZDY, 'mask', MASK) computes the
%   derivatives of the blocks projected onto DZDY. Note that MASK must
%   be specified in order to compute the derivative consistently with
%   the MASK in the forward pass. DZDX and DZDY have
%   the same dimesnions as X and Y respectivey.
%

% 2017 Hanh Tran based on spatial sparsity in a Winner Take All autoencoder
% writen follow the stype of vl_nndropout.m
opts.mask = [] ;

backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  opts = vl_argparse(opts, varargin) ;
end

if backMode && isempty(opts.mask)
  warning('vl_nnspatialSparsity: when using in backward mode, the mask should be specified') ;
end
if isempty(opts.mask)
    msk = zeros(size(x));
    if isa(x,'gpuArray')
        x_cpu = gather(x);
    else
        x_cpu = x;
    end
    x_max = max(max(x_cpu,[],1),[],2); %max value of each channel, size of 1 x 1 x nchannel x nSamp
    msk = bsxfun(@eq,x_cpu,x_max); % msk has same size with x_cpu     
% %     if length(size(x)) == 3 % 1 sample - TEST mode
% %         for i = 1:size(x,3)
% %             if isa(x,'gpuArray')
% %                 msk(:,:,i) = (gather(x(:,:,i)) == max(max(gather(x(:,:,i)))));
% %             else 
% %                 msk(:,:,i) = (x(:,:,i) == max(max(x(:,:,i))));
% %             end
% %         end
% %     else % 4 channels with batchSize samples - Train mode
% %         for j = 1:size(x,4)
% %             for i = 1:size(x,3)
% %                 if isa(x,'gpuArray')
% %                     msk(:,:,i,j) = (gather(x(:,:,i,j)) == max(max(gather(x(:,:,i,j)))));
% %                 else 
% %                     msk(:,:,i,j) = (x(:,:,i,j) == max(max(x(:,:,i,j))));
% %                 end
% %             end
% %         end
% %     end
  % product determines data type
  if isa(x,'gpuArray')
    opts.mask = gpuArray(msk);
  else
    opts.mask = msk ;
  end
end

% Apply mask. Note that mask is either `single` or `double`
% and a CPU or GPU array like the input argument `x`.
if ~backMode
  y = opts.mask .* x ;
else
  y = opts.mask .* dzdy ;
end
mask = opts.mask ;
