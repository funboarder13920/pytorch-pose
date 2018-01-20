addpath('jsonlab/')

% Download Synthetic hand dataset 
SYNTHROOTLOCAL = '../../hands_2017_compressed/data/synthetic/';

IMG = 'rgb_small_refined/';
IMG_LOCAL = 'rgb_small/';
SEGM = 'segm/';

addpath('jsonlab/')

count = 1;
train_perc = 0.8;

text = fileread('C:/Users/Valentin/Downloads/corres.json');
imgs = jsondecode(text);
corres = containers.Map(imgs{2,1}, imgs{1,1});

text = fileread('C:/Users/Valentin/Downloads/gen_2.json');
imgs = jsondecode(text);
imgs = imgs(randperm(size(imgs,1)));
nb_train = int32(train_perc*size(imgs,1));

% ---- train data
for im = 1:size(imgs,1)
    % trivial stuff for LEEDS
    if ~contains(imgs(im), '.png')
        continue
    end
    try
        path_this = strcat(SYNTHROOTLOCAL, IMG_LOCAL, corres(char(imgs(im))));
        [h,w,~] = size(imread(path_this));
    catch
       continue 
    end
    joint_all(count).dataset = 'SYNTH';
    joint_all(count).isValidation = 0 + 1*(im>nb_train);
    joint_all(count).img_paths = strcat(IMG, char(imgs(im)));
    joint_all(count).img_target_paths = strcat(SEGM, strrep(corres(char(imgs(im))),'.png', '.exr'));
    joint_all(count).annolist_index = count;

    joint_all(count).img_width = w;
    joint_all(count).img_height = h;

    count = count + 1;
    %fprintf('processing %s\n', path_this);
end

opt.FileName = strcat(SYNTHROOTLOCAL, 'synth_segm_refined_annotations.json');
opt.FloatFormat = '%.3f';
opt.Compact = 1;
savejson('', joint_all, opt);