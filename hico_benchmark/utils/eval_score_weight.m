function [ ap_ts ] = eval_score_weight( W, param_tr )

config;

% get parameters
feat_type    = param_tr.feat_type;
score_type   = param_tr.score_type;
score_dir_vb = param_tr.score_dir_vb;
score_dir_nn = param_tr.score_dir_nn;
score_dir_vo = param_tr.score_dir_vo;
coocc_act    = param_tr.coocc_act;
flag_obj     = param_tr.flag_obj;

% try loading cache file
res_file = ['res_obj' num2str(flag_obj) '_' strrep(score_type,'+','_') '.mat'];
res_file = [base_dir 'caches/svm_comb/mode_' feat_type '/' res_file];
if exist(res_file,'file')
    fprintf('result file loaded.\n');
    ld = load(res_file);
    ap_ts = ld.ap_ts;
    return
end

% load annotation
anno     = load(anno_file);
anno_sep = load(anno_sep_file);
anno_vb  = anno_sep.anno_vb;
anno_nn  = anno_sep.anno_nn;

num_class = numel(anno.list_action);

% initialize ap_ts
ap_ts = zeros(num_class,1);

fprintf('start test ...\n');
for i = 1:num_class
    % get labels
    ii    = anno.anno_test(i,:) ~= 0;
    label = anno.anno_test(i,ii);
    label(label == -2) = -1;

    % load scores
    if (strcmp(score_type,'v+o') == 1 ...
            || strcmp(score_type,'v+vo') == 1 ...
            || strcmp(score_type,'o+vo') == 1 ...
            || strcmp(score_type,'v+o+vo') == 1 ...
            || strcmp(score_type,'vo+coocc+v+o') == 1 ...
            || strcmp(score_type,'vo+coocc+v') == 1 ...
            || strcmp(score_type,'vo+coocc+o') == 1)
        % get im id for each vb and nn class
        vb_id = anno_vb.ind(i);
        nn_id = anno_nn.ind(i);
        im_id_vb = find(anno_vb.anno_test(vb_id,:) ~= 0);
        im_id_nn = find(anno_nn.anno_test(nn_id,:) ~= 0);
        assert(all(ismember(find(ii), im_id_vb)));
        assert(all(ismember(find(ii), im_id_nn)));
        % load vb and nn score
        vb_file = sprintf('%strain_test/verb_%d.mat',score_dir_vb,vb_id);
        nn_file = sprintf('%strain_test/noun_%d.mat',score_dir_nn,nn_id);
        ld_vb = load(vb_file);
        ld_nn = load(nn_file);
        score_vb = ld_vb.res.prob_estimates;
        score_nn = ld_nn.res.prob_estimates;
        score_vb = score_vb(ismember(im_id_vb, find(ii)));
        score_nn = score_nn(ismember(im_id_nn, find(ii)));
        assert(numel(score_vb) == numel(label));
        assert(numel(score_nn) == numel(label));
    end
    if (strcmp(score_type,'v+vo') == 1 ...
            || strcmp(score_type,'o+vo') == 1 ...
            || strcmp(score_type,'v+o+vo') == 1 ...
            || strcmp(score_type,'vo+coocc') == 1 ...
            || strcmp(score_type,'vo+coocc+v+o') == 1 ...
            || strcmp(score_type,'vo+coocc+v') == 1 ...
            || strcmp(score_type,'vo+coocc+o') == 1)
        % load vo score; never empty during test
        file_vo = sprintf('%sscore_%d.mat',score_dir_vo,i);
        ld = load(file_vo);
        score_vo = ld.score_ts(ii);
        assert(numel(score_vo) == numel(label));
    end
    if (strcmp(score_type,'vo+coocc') == 1 ...
            || strcmp(score_type,'vo+coocc+v+o') == 1 ...
            || strcmp(score_type,'vo+coocc+v') == 1 ...
            || strcmp(score_type,'vo+coocc+o') == 1)
        ii_svm = find(coocc_act(i,:) == 1);
        ii_svm(ii_svm == i) = [];  % remove class i
        score_coocc = [];
        for j = ii_svm
            % load vo score for co-occurred classes; skip the classes not 
            % using cv (during training)
            file_vo = sprintf('%sscore_%d.mat',score_dir_vo,j);
            ld = load(file_vo);
            if ~isempty(ld.score_tr)
                score_coocc = [score_coocc ld.score_ts(ii)];
                % % assertion
                % jj = anno.anno_test(j,:) ~= 0;
                % score_file_jnt = sprintf('%saction_%d.mat',score_base_jnt,j);
                % ld2 = load(score_file_jnt);
                % assert(sum(abs(ld.res_ts(jj~=0) - ld2.data.prob_estimates)) < 1e-2);
            else
                % do nothing: skip the classes not using cv
            end
        end
        assert(isempty(score_coocc) || size(score_coocc,1) == numel(label));
    end
    
    % concat scores
    switch score_type
        case 'v+o'
            score_phi = [score_vb score_nn];
        case 'v+vo'
            score_phi = [score_vb score_vo];
        case 'o+vo'
            score_phi = [score_nn score_vo];
        case 'v+o+vo'
            score_phi = [score_vb score_nn score_vo];
        case 'vo+coocc'
            score_phi = [score_vo score_coocc];
        case 'vo+coocc+v+o'
            score_phi = [score_vo score_coocc score_vb score_nn];
        case 'vo+coocc+v'
            score_phi = [score_vo score_coocc score_vb];
        case 'vo+coocc+o'
            score_phi = [score_vo score_coocc score_nn];
    end
    
    % manually change scores for the 'known object (Ko)' settings
    if flag_obj && ~isempty(score_phi)
        ii         = anno.anno_test(i,:) ~= 0;
        label_sneg = anno.anno_test(i,ii);
        score_phi(label_sneg == -2,:) = -1e10;
    end
    
    % compute ap
    switch score_type
        case {'v+o','v+vo','o+vo','v+o+vo'}
            score = W{i} * score_phi';
        case {'vo+coocc','vo+coocc+v+o','vo+coocc+v','vo+coocc+o'}
            score = W{i} * score_phi';
    end
    [~, ~, ap] = eval_pr_score_label(score, label, sum(label == 1), 0);
    
    ap_ts(i) = ap;
end
fprintf('done.\n\n');

% save to file
save(res_file,'ap_ts');

end

