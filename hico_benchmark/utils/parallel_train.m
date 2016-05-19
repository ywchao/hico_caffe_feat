function [  ] = parallel_train(start_index, end_index, label_type, feat_type, feat_dir)

config;

fprintf('start %d to %d\n', start_index, end_index);

switch label_type
    case {'vo'}
        anno = load(anno_file);
        model_temp = 'model_a_%d.mat';
        result_temp = 'action_%d.mat';
    case 'sep_vb'
        anno_sep = load(anno_sep_file);
        anno = anno_sep.anno_vb;
        model_temp = 'model_v_%d.mat';
        result_temp = 'verb_%d.mat';
    case 'sep_nn'
        anno_sep = load(anno_sep_file);
        anno = anno_sep.anno_nn;
        model_temp = 'model_n_%d.mat';
        result_temp = 'noun_%d.mat';
end

% set output dir
save_test_dir  = [base_dir 'caches/svm_' label_type '/mode_' feat_type '/train_test/'];
save_model_dir = [base_dir 'caches/svm_' label_type '/mode_' feat_type '/trained_model/'];
makedir(save_test_dir);
makedir(save_model_dir);
        
% start svm training and prediction parallelly
parfor j = start_index:end_index
    [model, ap, c_val, cvmodel, cvid, vscore] = train_one_action(anno, feat_type, feat_dir, j);
    model_name = sprintf(model_temp, j);
    save_model(fullfile(save_model_dir, model_name), model, ap, c_val, cvmodel, cvid, vscore);

    [label_value, predicted_label, accuracy, prob_estimates] = test_one_action(anno, feat_type, feat_dir, model, j);
    result_name = sprintf(result_temp, j);
    save_file = fullfile(save_test_dir, result_name);
    save_pred(save_file, label_value, predicted_label, accuracy, prob_estimates);
end

end
