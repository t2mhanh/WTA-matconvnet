clear all
close all
% Use data with size 240x360, patch size 48x48
%SHOULD TRY 158x238 (ped1) and 156x240 (ped2)

dataDir = {'./ucsd_data/FlowResize240x360/UCSDped1/Train' ...
    './ucsd_data/FlowResize240x360/UCSDped2/Train'};
imdbPath = './ucsd_data'
PatchData = './ucsd_data/OptPatches_48'; %save path
if ~exist(PatchData,'dir');mkdir(PatchData) ; else end

% A matfile contains paths to training optical flows size of 240x360
% it may help to chose training flows randomly
if exist(fullfile(imdbPath,'UCSDOptImdb.mat'))
  imdb = load(fullfile(imdbPath,'UCSDOptImdb.mat'));
else
  imdb = UCSDTrainDataSetup('dataDir', dataDir,'dataType','*.mat') ;
  save(imdbPath, '-struct', 'imdb') ;
end

%%
id = 1;
for i = 1:numel(imdb.images.id)
    i
        dataPath = sprintf(fullfile(imdb.images.path{i}, imdb.images.name{i}));
        load(dataPath);
        % crop original flow at foreground region
        mag = sum(Flow.^2,3);
        mask = mag(24:size(Flow,1)-24,24:size(Flow,2)-24) >= 0.5;% size 48x48
        [L,nr,np] = connectedcomponents(mask);
        for j = 1:nr
            L_randR = L == j;
            [r,c] = find(L_randR ==1);
            r_cen = floor((r(end) + r(1))./2);
            c_cen = floor((c(end) + c(1))./2);
            Flow_48 = single(Flow(r_cen:r_cen+47,c_cen:c_cen+47,:));
%             figure(1)
%             H_FlowVecVisual(Flow_48)
%             pause
%             Flow_48 = single(Flow(r_cen:r_cen+23,c_cen:c_cen+23,:));
            save(fullfile(PatchData,['Flow_48_' num2str(id) '.mat']),'Flow_48')
            id = id + 1;
        end
end
