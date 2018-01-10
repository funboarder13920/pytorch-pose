addpath('jsonlab/')

% Download Synthetic hand dataset 
SYNTHROOTLOCAL = '../../egohands_data/';

IMG = 'egohands/';
SEGM = 'egohands_segm/';

addpath('jsonlab/')

count = 1;
train_perc = 0.6;

imgs = dir([SYNTHROOTLOCAL IMG '*.png']);
imgs = imgs(randperm(size(imgs,1)));
nb_train = int32(train_perc*size(imgs,1));


% ---- train data
for im = 1:size(imgs,1)
    % trivial stuff for LEEDS
    joint_all(count).dataset = 'EGO';
    joint_all(count).isValidation = 0 + 1*(im>nb_train);
    joint_all(count).img_paths = strcat(IMG, imgs(im).name);
    joint_all(count).img_target_paths = strcat(SEGM, imgs(im).name);
    joint_all(count).annolist_index = count;

    path_this = strcat(SYNTHROOTLOCAL, IMG, imgs(im).name);
    [h,w,~] = size(imread(path_this));

    joint_all(count).img_width = w;
    joint_all(count).img_height = h;

    count = count + 1;
    %fprintf('processing %s\n', path_this);
end

opt.FileName = strcat(SYNTHROOTLOCAL, 'ego_segm_annotations.json');
opt.FloatFormat = '%.3f';
opt.Compact = 1;
savejson('', joint_all, opt);