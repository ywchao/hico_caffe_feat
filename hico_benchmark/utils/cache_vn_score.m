function [ res ] = cache_vn_score( anno_hico, action_id, anno_vb, anno_nn, feat_type, feat_dir )

config;

% load vb and nn models
model_base_vb = [base_dir 'caches/svm_sep_vb/mode_' feat_type '/trained_model/'];
model_base_nn = [base_dir 'caches/svm_sep_nn/mode_' feat_type '/trained_model/'];

vb_id = anno_vb.ind(action_id);
nn_id = anno_nn.ind(action_id);

model_vb_file = sprintf('%smodel_v_%d.mat',model_base_vb,vb_id);
model_nn_file = sprintf('%smodel_n_%d.mat',model_base_nn,nn_id);
assert(exist(model_vb_file,'file') ~= 0);
assert(exist(model_nn_file,'file') ~= 0);

model_vb = load(model_vb_file);
model_nn = load(model_nn_file);
model_vb = model_vb.res;
model_nn = model_nn.res;

% update cvid
im_id    = find(anno_hico.anno_train(action_id,:) ~= 0);
im_id_vb = find(anno_vb.anno_train(vb_id,:) ~= 0);
im_id_nn = find(anno_nn.anno_train(nn_id,:) ~= 0);
assert(all(ismember(im_id, im_id_vb)));
assert(all(ismember(im_id, im_id_nn)));

if ~isempty(model_vb.cvid)
    assert(numel(im_id_vb) == numel(model_vb.cvid));
    ii = ismember(im_id_vb, im_id);
    model_vb.cvid = model_vb.cvid(ii);
    % model_vb.vscore = model_vb.vscore(ii);
    fprintf('                  ');
else
    fprintf(' vb empty cvid ...');
end
if ~isempty(model_nn.cvid)
    assert(numel(im_id_nn) == numel(model_nn.cvid));
    ii = ismember(im_id_nn, im_id);
    model_nn.cvid = model_nn.cvid(ii);
    % model_nn.vscore = model_nn.vscore(ii);
else
    error(' nn empty cvid ... this should not happen\n');
end

% process training
% TODO: bug, should not run any svm predict for the training data
%  1. w cv:    use vscore
%  2. w/o cv:  just leave empty; those values are not used in learn_score_weights.m already
anno   = anno_hico.anno_train(action_id,:);
list   = anno_hico.list_train;
mode   = 1;
res_tr = get_vn_one_set(anno, list, model_vb, model_nn, feat_type, feat_dir, mode);

% process test
anno   = anno_hico.anno_test(action_id,:);
list   = anno_hico.list_test;
mode   = 2;
res_ts = get_vn_one_set(anno, list, model_vb, model_nn, feat_type, feat_dir, mode);

res.res_tr = res_tr;
res.res_ts = res_ts;

end


function [ res ] = get_vn_one_set( anno, list, model_vb, model_nn, feat_type, feat_dir, mode )
% mode
%   1: training
%   2: test

config;

% load mean norm file
mean_norm_file = [base_dir 'caches/mean_norm/' feat_type '.mat'];
ld = load(mean_norm_file);
mean_norm = ld.mean_norm;

% get labels
indexes     = find(anno);
label_value = anno(indexes)';
label_value(label_value == -2) = -1;

% get feature files
images_name = list(indexes);
feature_list = cellfun(@(x)strrep(x,'.jpg','.mat'), images_name, 'UniformOutput', false);

% check input
if mode == 1
    assert(isempty(model_vb.cvid) || numel(label_value) == numel(model_vb.cvid));
    assert(isempty(model_nn.cvid) || numel(label_value) == numel(model_nn.cvid));
end

% get scores
res.label    = label_value;
res.vb_pred  = zeros(length(feature_list),1);
res.vb_score = zeros(length(feature_list),1);
res.nn_pred  = zeros(length(feature_list),1);
res.nn_score = zeros(length(feature_list),1);

for j = 1:length(feature_list)
    % load feature
    s = load(fullfile(feat_dir, feature_list{j}));
    % Normalization factor is picked by the average features norm
    % during training.
    data = double(s.feat) ./ mean_norm;

    % vb prediction
    [model, ~]   = get_model(model_vb, mode, j);
    % [pred, ~, score] = predict(label_value(j), sparse(data), model, '-b 0');
    [~, pred, ~, score] = evalc('predict(label_value(j), sparse(data), model, ''-b 0'');');
    res.vb_pred(j)  = pred;
    res.vb_score(j) = score;
    % if mode == 1 && ~isempty(model_vb.cvid)
    %     assert(score == model_vb.vscore(j));
    % end
    
    % nn prediction
    [model, ~]   = get_model(model_nn, mode, j);
    % [pred, ~, score] = predict(label_value(j), sparse(data), model, '-b 0');
    [~, pred, ~, score] = evalc('predict(label_value(j), sparse(data), model, ''-b 0'');');
    res.nn_pred(j)  = pred;
    res.nn_score(j) = score;
    % if mode == 1 && ~isempty(model_nn.cvid)
    %     assert(score == model_nn.vscore(j));
    % end
end

% set is_cv
switch mode
    case 1
        res.vb_is_cv = isempty(model_vb.cvid) == 0;
        res.nn_is_cv = isempty(model_nn.cvid) == 0;
    case 2
        res.vb_is_cv = NaN;
        res.nn_is_cv = NaN;
end

end


function [ model, is_cv ] = get_model( model_all, mode, j )
% get model

switch mode
    case 1
        if isempty(model_all.cvid)
            % TODO: bug, should fix this!
            model    = model_all.model;
            is_cv    = 0;
        else
            % TODO: bug, should fix this!
            model_id = model_all.cvid(j);
            model    = model_all.cvmodel{model_id};
            is_cv    = 1;
        end
    case 2
        model = model_all.model;
        is_cv = NaN;
end

end
