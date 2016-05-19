
config;

% set label type

% label_type  = 'sep_vb';
% label_type  = 'sep_nn';

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

% support Parallelization by batching VB/NN classes
num_batch = 1;
batch_id  = 1;

anno_sep = load(anno_sep_file);
switch label_type
    case 'sep_vb'
        len = length(anno_sep.anno_vb.list);
    case 'sep_nn'
        len = length(anno_sep.anno_nn.list);
end
interval  = round(len / num_batch);
ss        = 1:interval:len;
sid       = ss(1:num_batch);
eid       = [ss(2:num_batch)-1 len];

% start svm training and prediction
fprintf('start training %s classifier ... \n',label_type);
fprintf('feat_type: %s\n',feat_type);
fprintf('num_batch: %03d\n',num_batch);
fprintf('batch_id:  %03d\n',batch_id);
fprintf('num_class: %03d\n\n',eid(batch_id)-sid(batch_id)+1);

parallel_train(sid(batch_id), eid(batch_id), label_type, feat_type, feat_dir);

fprintf('\ndone training %s classifiers.\n\n',label_type);

% delete parpool
delete(poolobj);
