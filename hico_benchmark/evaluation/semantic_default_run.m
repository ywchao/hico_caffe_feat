
config;

% set feature type
%   'imagenet' is the default. You can select another feature type by
%   uncommenting the line.

% feat_type = 'imagenet';
% feat_type = 'ft_verb';
% feat_type = 'ft_object';
% feat_type = 'ft_action';

% set parfor option
%   set to false for exact reproduction of iccv result

% use_parfor = true;
% use_parfor = false;

% load annotation
anno = load(anno_file);

% start evaluation
fprintf('start evaluation\n');
fprintf('setting:    default\n');
fprintf('feat_type:  %s\n',feat_type);
fprintf('num_class:  %03d\n',numel(anno.list_action));
fprintf('use_parfor: %d\n\n',use_parfor);

% init param_tr
param_tr = [];
param_tr.feat_type = feat_type;

% evaluate with the default setting
param_tr.flag_obj = false;

% load co-occurence map
coocc_data = load(coocc_file);
param_tr.coocc_act = coocc_data.coocc_act;

% set directory
param_tr.score_dir_vb = [base_dir 'caches/svm_sep_vb/mode_' feat_type '/'];
param_tr.score_dir_nn = [base_dir 'caches/svm_sep_nn/mode_' feat_type '/'];
param_tr.score_dir_vo = [base_dir 'caches/svm_comb/mode_' feat_type '/vo_score_all_im/'];

% set parfor option
param_tr.use_parfor = use_parfor;

% start parpool
if param_tr.use_parfor
    if ~exist('poolsize','var')
        poolobj = parpool();
    else
        poolobj = parpool(poolsize);
    end
end

% evaluate vo
res_vo = eval_vo(feat_type, param_tr.flag_obj);

% evaluate v+o
param_tr.score_type = 'v+o';
W = train_score_weight(param_tr);
res_v_o = eval_score_weight(W, param_tr);

% evaluate v+vo
param_tr.score_type = 'v+vo';
W = train_score_weight(param_tr);
res_v_vo = eval_score_weight(W, param_tr);

% evaluate o+vo
param_tr.score_type = 'o+vo';
W = train_score_weight(param_tr);
res_o_vo = eval_score_weight(W, param_tr);

% evaluate v+o+vo
param_tr.score_type = 'v+o+vo';
W = train_score_weight(param_tr);
res_v_o_vo = eval_score_weight(W, param_tr);

% evaluate vo+coocc
param_tr.score_type = 'vo+coocc';
W = train_score_weight(param_tr);
res_vo_coocc = eval_score_weight(W, param_tr);

% evaluate vo+coocc+v
param_tr.score_type = 'vo+coocc+v';
W = train_score_weight(param_tr);
res_vo_coocc_v = eval_score_weight(W, param_tr);

% evaluate vo+coocc+o
param_tr.score_type = 'vo+coocc+o';
W = train_score_weight(param_tr);
res_vo_coocc_o = eval_score_weight(W, param_tr);

% evaluate vo+coocc+v+o
param_tr.score_type = 'vo+coocc+v+o';
W = train_score_weight(param_tr);
res_vo_coocc_v_o = eval_score_weight(W, param_tr);

% delete parpool
if param_tr.use_parfor
    delete(poolobj);
end

% print evaluation results
fprintf('evaluation done.\n');
fprintf('----------------\n');
fprintf('setting:   default\n');
fprintf('feat_type: %s\n',feat_type);
fprintf('num_class: %03d\n\n',numel(anno.list_action));
fprintf('mAP (full/rare):\n');

ii = find(sum(anno.anno_train==1,2) < 5);
fprintf('  VO:            %6.2f / % 6.2f\n', mean(res_vo(:,6)) * 100, mean(res_vo(ii,6)) * 100);
fprintf('  V+O:           %6.2f / % 6.2f\n', mean(res_v_o) * 100, mean(res_v_o(ii)) * 100);
fprintf('  V+VO:          %6.2f / % 6.2f\n', mean(res_v_vo) * 100, mean(res_v_vo(ii)) * 100);
fprintf('  O+VO:          %6.2f / % 6.2f\n', mean(res_o_vo) * 100, mean(res_o_vo(ii)) * 100);
fprintf('  V+O+VO:        %6.2f / % 6.2f\n', mean(res_v_o_vo) * 100, mean(res_v_o_vo(ii)) * 100);
fprintf('  VO+coocc:      %6.2f / % 6.2f\n', mean(res_vo_coocc) * 100, mean(res_vo_coocc(ii)) * 100);
fprintf('  VO+coocc+V:    %6.2f / % 6.2f\n', mean(res_vo_coocc_v) * 100, mean(res_vo_coocc_v(ii)) * 100);
fprintf('  VO+coocc+O:    %6.2f / % 6.2f\n', mean(res_vo_coocc_o) * 100, mean(res_vo_coocc_o(ii)) * 100);
fprintf('  VO+coocc+V+O:  %6.2f / % 6.2f\n', mean(res_vo_coocc_v_o) * 100, mean(res_vo_coocc_v_o(ii)) * 100);
