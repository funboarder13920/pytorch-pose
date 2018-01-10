addpath('jsonlab/')

% Download Synthetic hand dataset 
REALROOTLOCAL = '../../GRASP/';
GUN = 'GRASP/';

addpath('jsonlab/')

count = 1;

% ---- train data
first_dirs = dir(REALROOTLOCAL);
for first_dir_it = 1:size(first_dirs, 1)
    if strcmp(first_dirs(first_dir_it).name ,'..') || strcmp(first_dirs(first_dir_it).name, '.') || ~first_dirs(first_dir_it).isdir
        continue
    end
    
    second_dirs = dir([REALROOTLOCAL first_dirs(first_dir_it).name]);
    for second_dir_it = 1:size(second_dirs, 1)
        if strcmp(second_dirs(second_dir_it).name, '..') || strcmp(second_dirs(second_dir_it).name, '.') || ~second_dirs(second_dir_it).isdir
            continue
        end
        
        imgs = dir([second_dirs(second_dir_it).folder '\' second_dirs(second_dir_it).name '\*.jpg']);
        imgs = imgs(randperm(size(imgs,1)));
        for im = 1:size(imgs,1)
            % trivial stuff for LEEDS
            joint_all(count).dataset = 'GUN';
            joint_all(count).isValidation = 1;
            joint_all(count).img_paths = strcat(GUN, first_dirs(first_dir_it).name, '/', second_dirs(second_dir_it).name, '/',imgs(im).name);
            joint_all(count).img_target_paths = '';
            joint_all(count).annolist_index = count;

            path_this = strcat(REALROOTLOCAL, first_dirs(first_dir_it).name, '/', second_dirs(second_dir_it).name, '/',imgs(im).name);
            [h,w,~] = size(imread(path_this));

            joint_all(count).img_width = w;
            joint_all(count).img_height = h;

            count = count + 1;
            %fprintf('processing %s\n', path_this);
        end
    end
end

opt.FileName = strcat(REALROOTLOCAL, 'synth_segm_gun_annotations.json');
opt.FloatFormat = '%.3f';
opt.Compact = 1;
savejson('', joint_all, opt);