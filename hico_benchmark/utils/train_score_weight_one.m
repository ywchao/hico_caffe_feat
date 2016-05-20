function [ W, ap_tr ] = train_score_weight_one( i, num_class, act_name, anno, anno_vb, anno_nn, w_cand_all, param_tr )

% get parameters
score_type   = param_tr.score_type;
score_dir_vb = param_tr.score_dir_vb;
score_dir_nn = param_tr.score_dir_nn;
score_dir_vo = param_tr.score_dir_vo;
coocc_act    = param_tr.coocc_act;
flag_obj     = param_tr.flag_obj;

fprintf('  %03d/%03d %-29s   ',i,num_class,act_name);

% get labels
ii    = anno.anno_train(i,:) ~= 0;
label = anno.anno_train(i,ii);
label(label == -2) = -1;

% load scores
is_cv_sep = true;
is_cv_vo  = true;
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
    im_id_vb = find(anno_vb.anno_train(vb_id,:) ~= 0);
    im_id_nn = find(anno_nn.anno_train(nn_id,:) ~= 0);
    assert(all(ismember(find(ii), im_id_vb)));
    assert(all(ismember(find(ii), im_id_nn)));
    % load vb and nn score
    vb_file = sprintf('%strained_model/model_v_%d.mat',score_dir_vb,vb_id);
    nn_file = sprintf('%strained_model/model_n_%d.mat',score_dir_nn,nn_id);
    ld_vb = load(vb_file);
    ld_nn = load(nn_file);
    score_vb = ld_vb.res.vscore;
    score_nn = ld_nn.res.vscore;
    if ~isempty(score_vb)
        score_vb = score_vb(ismember(im_id_vb, find(ii)));
        assert(numel(score_vb) == numel(label));
    end
    if ~isempty(score_nn)
        score_nn = score_nn(ismember(im_id_nn, find(ii)));
        assert(numel(score_nn) == numel(label));
    end
    % set is_cv_sep to false if not using cv for either vb or nn
    is_cv_sep = ~(isempty(score_vb) || isempty(score_nn));
end
if (strcmp(score_type,'v+vo') == 1 ...
        || strcmp(score_type,'o+vo') == 1 ...
        || strcmp(score_type,'v+o+vo') == 1 ...
        || strcmp(score_type,'vo+coocc') == 1 ...
        || strcmp(score_type,'vo+coocc+v+o') == 1 ...
        || strcmp(score_type,'vo+coocc+v') == 1 ...
        || strcmp(score_type,'vo+coocc+o') == 1)
    % load vo score
    vo_file = sprintf('%sscore_%d.mat',score_dir_vo,i);
    ld = load(vo_file);
    if ~isempty(ld.score_tr)
        score_vo = ld.score_tr(ii);
        is_cv_vo = true;
    else
        score_vo = [];
        is_cv_vo = false;
    end
    assert(isempty(score_vo) || numel(score_vo) == numel(label));
end
if (strcmp(score_type,'vo+coocc') == 1 ...
        || strcmp(score_type,'vo+coocc+v+o') ...
        || strcmp(score_type,'vo+coocc+v') == 1 ...
        || strcmp(score_type,'vo+coocc+o') == 1)
    ii_svm = find(coocc_act(i,:) == 1);
    ii_svm(ii_svm == i) = [];  % ignore class i
    score_coocc = [];
    for j = ii_svm
        % load vo score for co-occurred classes
        vo_file = sprintf('%sscore_%d.mat',score_dir_vo,j);
        ld = load(vo_file);
        if ~isempty(ld.score_tr)
            score_coocc = [score_coocc ld.score_tr(ii)];
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

% manually change scores for the 'known object (KO)' settings
if flag_obj && ~isempty(score_phi)
    ii         = anno.anno_train(i,:) ~= 0;
    label_sneg = anno.anno_train(i,ii);
    score_phi(label_sneg == -2,:) = -1e10;
end

% solve weights
if is_cv_sep && is_cv_vo
    % run grid search on weights
    w_cand = w_cand_all{size(score_phi,2)};
    ap_t   = zeros(size(w_cand,1),1);
    score  = w_cand * score_phi';
    for j = 1:size(w_cand, 1)
        [~, ~, ap] = eval_pr_score_label(score(j,:), label, sum(label == 1), 0, true);
        ap_t(j) = ap;
    end
    
    ii = find(ap_t == max(ap_t));
    pp = randperm(numel(ii));
    ii = ii(pp(1));
    
    W     = w_cand(ii,:);
    ap_tr = ap_t(ii);
else
    % set uniform weight if a subset of cv scores is missing
    switch score_type
        case {'v+o','v+vo','o+vo'}
            nn = 2;
        case 'v+o+vo'
            nn = 3;
        case 'vo+coocc'
            nn = size(score_coocc,2)+1;
        case {'vo+coocc+v','vo+coocc+o'}
            nn = size(score_coocc,2)+2;
        case 'vo+coocc+v+o'
            nn = size(score_coocc,2)+3;
    end
    W = repmat(1/nn,[1 nn]);
    ap_tr = NaN;
end

% display
switch score_type
    case {'v+o','v+vo','o+vo'}
        fprintf('w1: %4.2f  w2: %4.2f   ap: %6.2f\n',W(1),W(2),100*ap_tr);
    case 'v+o+vo'
        fprintf('w1: %4.2f  w2: %4.2f  w3: %5.3f   ap: %6.2f\n',W(1),W(2),W(3),100*ap_tr);
    case {'vo+coocc','vo+coocc+v+o','vo+coocc+v','vo+coocc+o'}
        for j = 1:7
            if j <= numel(W)
                fprintf('w%d: %4.2f  ',j,W(j));
            else
                fprintf('          ');
            end
        end
        fprintf(' ap: %6.2f\n',100*ap_tr);
end

end
