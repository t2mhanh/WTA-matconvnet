function imdb = UCSDTrainDataSetup(varargin)
% Train data
% dataDir: Train folder (contains .dataType files)
opts.dataDir = {'./ucsd_data/FlowResize240x360/UCSDped1/Train' ...
    './ucsd_data/FlowResize240x360/UCSDped2/Train'};
opts.dataType = '*.mat';

opts = vl_argparse(opts, varargin) ;
imdb.sets.id = uint8([1 2 3]) ;
imdb.sets.name = {'train', 'val', 'test'} ;
imdb.images.name = {};
imdb.images.set = [];
imdb.images.id = [];
imdb.images.path = {};
j = length(imdb.images.id);

for numdir = 1:numel(opts.dataDir)
    imagePath = {};
    if length(dir(fullfile(opts.dataDir{numdir},opts.dataType))) > 0 ; imagePath{1} = opts.dataDir{numdir};
    else
        directory = dir(fullfile(opts.dataDir{numdir}));
        directory = directory([directory.isdir]);
        dirId = 1;
        for Id = 1:length(directory)
            if length(dir(fullfile(opts.dataDir{numdir},directory(Id).name,opts.dataType)))>0;
                imagePath{dirId} = fullfile(opts.dataDir{numdir},directory(Id).name);
                dirId = dirId + 1;
            else end
        end
    end

    for Id = 1:numel(imagePath)
        data = dir(fullfile(imagePath{Id},opts.dataType));
        totalData = length(data(not([data.isdir])));
        for i = 1:totalData
            j = j + 1;
            imdb.images.id(j) = j;
            imdb.images.name{j} = data(i).name;
            imdb.images.path{j} = imagePath{Id};
            imdb.images.set(j) = 1;
        end
    end

end

val = randsample(length(imdb.images.id),100); % 100 samples for validation. Enough?
for i = 1 :length(val)
    imdb.images.set(val(i)) = 2;
end

imdb.images.id = uint32(imdb.images.id) ;
imdb.images.set = uint8(imdb.images.set) ;
