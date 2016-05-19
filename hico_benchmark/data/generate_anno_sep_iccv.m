
config;

if ~exist(anno_sep_file,'file')
    fprintf('generating anno_iccv_sep.mat (annotations for verbs and nouns) ... \n');

    anno = load(anno_file);
    [anno_vb, anno_nn] = convert_anno_vn_iccv(anno);
    save(anno_sep_file,'anno_vb','anno_nn');

    % print stats
    if 0
        vb_pos = sum(anno_vb.anno_train ==  1,2);
        vb_neg = sum(anno_vb.anno_train == -1,2);
        [~,ii] = sort(vb_pos,'descend');
        [{anno_vb.list(ii).vname_ing}' num2cell(vb_pos(ii)) num2cell(vb_neg(ii))]
        
        nn_pos = sum(anno_nn.anno_train ==  1,2);
        nn_neg = sum(anno_nn.anno_train == -1,2);
        [~,ii] = sort(nn_pos,'descend');
        [{anno_nn.list(ii).nname}' num2cell(nn_pos(ii)) num2cell(nn_neg(ii))]
    end

    fprintf('done.\n\n');
end
