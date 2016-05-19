
config;

% set feature type
%   'imagenet' is the default. You can select another feature type by
%   uncommenting the line.

% feat_type = 'imagenet';
% feat_type = 'ft_verb';
% feat_type = 'ft_object';
% feat_type = 'ft_action';

% set feature directory
% feat_dir = [base_dir './data/precomputed_dnn_features/' feat_type '/'];
feat_dir = [base_dir '../output/' strrep(feat_type ,'-','/') '/'];

% get mean norm on training set
mean_norm_base = [base_dir 'caches/mean_norm/'];
makedir(mean_norm_base);

mean_norm_file = [mean_norm_base feat_type '.mat'];
if ~exist(mean_norm_file,'file')
    fprintf('computing mean norm ... \n');
    mean_norm = get_mean_norm(feat_dir);
    save(mean_norm_file,'mean_norm');
else
    fprintf('loading mean norm ... \n');
    ld = load(mean_norm_file);
    mean_norm = ld.mean_norm;
end
fprintf('mean norm = %6.2f',mean_norm);
fprintf('\n\n');

% start parpool
if ~exist('poolsize','var')
    poolobj = parpool();
else
    poolobj = parpool(poolsize);
end

% support Parallelization by batching HOI classes
num_batch = 1;
batch_id  = 1;

anno      = load(anno_file);
len       = numel(anno.list_action);
interval  = round(len / num_batch);
ss        = 1:interval:len;
sid       = ss(1:num_batch);
eid       = [ss(2:num_batch)-1 len];

% start caching score
fprintf('start caching score ... \n');
fprintf('feat_type: %s\n',feat_type);
fprintf('num_batch: %03d\n',num_batch);
fprintf('batch_id:  %03d\n',batch_id);
fprintf('num_class: %03d\n\n',eid(batch_id)-sid(batch_id)+1);

start_index = sid(batch_id);
end_index   = eid(batch_id);

% set output dir
save_dir_vn = [base_dir 'caches/svm_comb/mode_' feat_type '/vn_score/'];
save_dir_vo = [base_dir 'caches/svm_comb/mode_' feat_type '/vo_score_all_im/'];
makedir(save_dir_vn);
makedir(save_dir_vo);

% load anno_vb and anno_nn
anno_sep = load(anno_sep_file);
anno_vb  = anno_sep.anno_vb;
anno_nn  = anno_sep.anno_nn;

parfor j = start_index:end_index
    fprintf('processing class %03d/%03d ...',j,len);
    tot_th = tic;
    
    % cache vn scores
    save_file = sprintf('%sscore_%d.mat', save_dir_vn, j);
    if ~exist(save_file,'file')
        res = cache_vn_score(anno, j, anno_vb, anno_nn, feat_type, feat_dir);
        save_score(save_file, res);
    end
    
    % cache vo scores for all training/test images
    save_file = sprintf('%sscore_%d.mat', save_dir_vo, j);
    if ~exist(save_file,'file')
        res = cache_vo_score_all_im(anno, j, feat_type, feat_dir);
        save_score(save_file, res);
    end
    
    fprintf('  done  %7.3f sec\n',toc(tot_th));
end

fprintf('\ndone caching score.\n\n');

% delete parpool
delete(poolobj);
