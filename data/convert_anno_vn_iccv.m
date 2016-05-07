function [ anno_vb, anno_nn ] = convert_anno_vn( anno )
% Note: 
%   We separate the data at the word-level instead of Wordnet
%   sense-level since
%     1. Some verb phrases do not have corresponding wordnet synsets,
%        i.e. intransitive verbs + prep
%
%   In this work, the Wordnet synsets are used mainly for giving
%   verb/noun definitions in the annnotation tasks
%


% get unique verbs
[~, ia, ic] = unique({anno.list_action.vname_ing}');

list = anno.list_action(ia);
list = rmfield(list,{'nname'});

% get verb annotations
[anno_train, anno_test] = convert_anno(anno, ic);

anno_vb.list        = list;
anno_vb.ind         = ic;
anno_vb.anno_train  = anno_train;
anno_vb.anno_test   = anno_test;
anno_vb.list_train  = anno.list_train;
anno_vb.list_test   = anno.list_test;


% get unique nounts
[~, ia, ic] = unique({anno.list_action.nname}');

list = anno.list_action(ia);
list = rmfield(list,{'vname','vname_ing','syn','def','synset'});
if isfield(list,'add_def')
    list = rmfield(list,{'add_def'});
end

% get verb annotations
[anno_train, anno_test] = convert_anno(anno, ic);

anno_nn.list        = list;
anno_nn.ind         = ic;
anno_nn.anno_train  = anno_train;
anno_nn.anno_test   = anno_test;
anno_nn.list_train  = anno.list_train;
anno_nn.list_test   = anno.list_test;


end


function [ anno_train, anno_test ] = convert_anno( anno, ind )

anno_train = zeros(numel(unique(ind)),size(anno.anno_train,2));
anno_test  = zeros(numel(unique(ind)),size(anno.anno_test,2));

for i = 1:numel(unique(ind))
    ii = ind == i;
    
    ii_pos = any(anno.anno_train(ii,:) == 1,1);
    ii_neg = ~any(anno.anno_train(ii,:) == 1,1) ...
        & (any(anno.anno_train(ii,:) == -1 | anno.anno_train(ii,:) == -2,1));  % Should this be all instead of any (matters only for vb)? any is used here due to iccv anno.
    assert(all((ii_pos & ii_neg) == 0));
    assert(sum(all(anno.anno_train(ii,:) == 0,1)) == (size(anno_train,2) - sum(ii_pos) - sum(ii_neg)));
    anno_train(i, ii_pos) = 1;
    anno_train(i, ii_neg) = -1;
    
    ii_pos = any(anno.anno_test(ii,:) == 1,1);
    ii_neg = ~any(anno.anno_test(ii,:) == 1,1) ...
        & (any(anno.anno_test(ii,:) == -1 | anno.anno_test(ii,:) == -2,1));  % Should this be all instead of any (matters only for vb)? any is used here due to iccv anno.
    assert(all((ii_pos & ii_neg) == 0));
    assert(sum(all(anno.anno_test(ii,:) == 0,1)) == (size(anno_test,2) - sum(ii_pos) - sum(ii_neg)));
    anno_test(i, ii_pos) = 1;
    anno_test(i, ii_neg) = -1;
end

end

