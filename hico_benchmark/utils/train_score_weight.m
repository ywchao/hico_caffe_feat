function [ W, ap_tr ] = train_score_weight( param_tr )

config;

% get parameters
feat_type  = param_tr.feat_type;
score_type = param_tr.score_type;
flag_obj   = param_tr.flag_obj;
use_parfor = param_tr.use_parfor;

% try loading cache file
w_file = ['w_obj' num2str(flag_obj) '_' strrep(score_type,'+','_') '.mat'];
w_file = [base_dir 'caches/svm_comb/mode_' feat_type '/' w_file];
if exist(w_file,'file')
    fprintf('trained file loaded.\n');
    ld = load(w_file);
    W = ld.W;
    ap_tr = ld.ap_tr;
    return
end

% load annotation
anno = load(anno_file);

num_class = numel(anno.list_action);
act_name  = cellfun(@(x,y)[x ' ' y], ...
    {anno.list_action.nname}', ...
    {anno.list_action.vname_ing}', ...
    'Uniformoutput',false ...
    );

% initialize W and ap_tr
W     = cell(num_class,1);
ap_tr = zeros(num_class,1);

% get candidate weights
w_cand_all = cell(7,1);
for i = 1:7
    w_cand_all{i} = get_w_cand(i);
end

% start timer
tic;

fprintf('score_type: %s\n',score_type);
fprintf('start training weights ... \n');
if use_parfor
    % use parfor loop
    % run rseed at the start of each iteration to ensure reproducibility
    parfor i = 1:num_class
        rseed;
        [W{i}, ap_tr(i)] = train_score_weight_one(i, num_class, act_name{i}, anno, param_tr, w_cand_all);
    end
else
    % use regular for loop
    % run rseed before the loop to ensure reproducibility; required for
    % exact reproduction of the iccv result
    rseed;
    for i = 1:num_class
        [W{i}, ap_tr(i)] = train_score_weight_one(i, num_class, act_name{i}, anno, param_tr, w_cand_all);
    end
end
fprintf('done. running time is %.2f sec.\n',toc);

% save to file
save(w_file,'W','ap_tr');

end
