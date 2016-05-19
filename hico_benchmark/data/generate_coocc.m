
config;

if ~exist(coocc_file,'file')
    fprintf('generating coocc.mat ... \n');
    
    anno = load(anno_file);
    anno_all   = [anno.anno_train anno.anno_test];
    list_nname = unique({anno.list_action.nname}');
    
    % generate normalized co-ocurrence map
    thres = 0.5;
    coocc     = cell(numel(list_nname),1);
    coocc_th  = cell(numel(list_nname),1);
    coocc_act = zeros(numel(anno.list_action),numel(anno.list_action));
    coocc_all = zeros(numel(anno.list_action),numel(anno.list_action));
    for i = 1:numel(list_nname)
        nname = list_nname{i};
        ii = cell_find_string({anno.list_action.nname}',nname);
        anno_all_t = anno_all(ii,:);
        anno_all_t(anno_all_t == -1) = 0;
        anno_all_t(anno_all_t == -2) = 0;
        coocc{i} = anno_all_t * anno_all_t';
        coocc{i} = coocc{i} ./ repmat(diag(coocc{i}),[1 size(coocc{i},2)]);
        
        coocc_th{i} = coocc{i};
        coocc_th{i}(coocc_th{i} > thres)  = 1;
        coocc_th{i}(coocc_th{i} <= thres) = 0;
        
        coocc_act(ii,ii) = coocc_th{i};
        coocc_all(ii,ii) = 1;
    end
    
    % save to file
    pathstr = fileparts(coocc_file);
    if ~exist(pathstr,'dir')
        makedir(pathstr);
    end
    save(coocc_file,'coocc','coocc_th','coocc_act','coocc_all');
    
    fprintf('done.\n\n');
end

% TODO: add visualization
