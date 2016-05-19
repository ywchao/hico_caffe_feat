function [ res ] = cache_vo_score_all_im( anno_hico, action_id, feat_type, feat_dir )

config;

% load trained model
model_base = [base_dir 'caches/svm_vo/mode_' feat_type '/trained_model/'];

model_file = sprintf('%smodel_a_%d.mat',model_base,action_id);
assert(exist(model_file,'file') ~= 0);

model = load(model_file);
model = model.res;

% cache training score
if isempty(model.vscore)
    score_tr = [];
else
    anno   = anno_hico.anno_train(action_id,:);
    list   = anno_hico.list_train;
    res_tr = get_score_ignored(anno, list, model, feat_type, feat_dir);
    
    score_tr = zeros(numel(anno_hico.list_train),1);
    score_tr(anno_hico.anno_train(action_id,:) ~= 0) = model.vscore;
    score_tr(anno_hico.anno_train(action_id,:) == 0) = res_tr.score;
    
    assert(~any(score_tr == 0));
end

% load test score
test_base = [base_dir 'caches/svm_vo/mode_' feat_type '/train_test/'];
test_file = sprintf('%saction_%d.mat',test_base,action_id);
ld    = load(test_file);
score = ld.res.prob_estimates;

% cache test score
anno   = anno_hico.anno_test(action_id,:);
list   = anno_hico.list_test;
res_ts = get_score_ignored(anno, list, model, feat_type, feat_dir);

score_ts = zeros(numel(anno_hico.list_test),1);
score_ts(anno_hico.anno_test(action_id,:) ~= 0) = score;
score_ts(anno_hico.anno_test(action_id,:) == 0) = res_ts.score;

assert(~any(score_ts == 0));

% save score
res.score_tr = score_tr;
res.score_ts = score_ts;

end


function [ res ] = get_score_ignored( anno, list, model, feat_type, feat_dir )
% mode
%   1: training
%   2: test

config;

% load mean norm file
mean_norm_file = [base_dir 'caches/mean_norm/' feat_type '.mat'];
ld = load(mean_norm_file);
mean_norm = ld.mean_norm;

% get feature files of ignored images
images_name = list(anno == 0);
feature_list = cellfun(@(x)strrep(x,'.jpg','.mat'), images_name, 'UniformOutput', false);

% get scores
res.pred  = zeros(length(feature_list),1);
res.score = zeros(length(feature_list),1);

for j = 1:length(feature_list)
    % load feature
    s = load(fullfile(feat_dir, feature_list{j}));
    % Normalization factor is picked by the average features norm
    % during training.
    data = double(s.feat) ./ mean_norm;
    
    % prediction
    % [pred, ~, score] = predict(0, sparse(data), model.model, '-b 0');
    [~, pred, ~, score] = evalc('predict(0, sparse(data), model.model, ''-b 0'');');
    res.pred(j)  = pred;
    res.score(j) = score;
end

end
