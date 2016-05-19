function [  ] = ft_prepare_patches( label_type, op_type )

config;

% set path
cache_base = [cache_root 'ft_' label_type '_' op_type '/'];

% make directory
makedir(cache_base);

% set paramters
rsize = [256 256];
is_flip = false;

ip_file_tr  = [cache_base 'train.txt'];
lmdb_dir_tr = [cache_base 'hico_train_lmdb'];
lmdb_dir_vl = [cache_base 'hico_val_lmdb'];

if ~exist(ip_file_tr,'file')
    % load annotation
    fprintf('loading original images ... \n');
    switch label_type
        case 'action'
            anno = load(anno_file);
            [train_imdata, train_labels, is_skip] = load_imdata_label( ...
                anno.list_action, ...
                anno.list_train, ...
                anno.anno_train, ...
                im_root, ...
                op_type, ...
                is_flip);
        case 'verb'
            load(anno_sep_file);
            [train_imdata, train_labels, is_skip] = load_imdata_label( ...
                anno_vb.list, ...
                anno_vb.list_train, ...
                anno_vb.anno_train, ...
                im_root, ...
                op_type, ...
                is_flip);
        case 'object'
            load(anno_sep_file);
            [train_imdata, train_labels, is_skip] = load_imdata_label( ...
                anno_nn.list, ...
                anno_nn.list_train, ...
                anno_nn.anno_train, ...
                im_root, ...
                op_type, ...
                is_flip);
    end
    assert(all(is_skip == 0));
    fprintf('done.\n');
    
    % shuffle
    rseed;
    perm_idx     = randperm(numel(train_imdata));
    train_imdata = train_imdata(perm_idx);
    train_labels = train_labels(perm_idx);

    im_path = {train_imdata.im_path}';
    label   = {train_labels.label}';
    assert(numel(im_path) == numel(im_path));
    
    % % make labels start from 0
    % label   = cellfun(@(x){x-1},label);
    
    % write to input file
    fprintf('generating lmdb input file ... \n');
    fid = fopen(ip_file_tr,'w');
    for i = 1:numel(im_path)
        tic_print(sprintf('  %05d/%05d\n',i,numel(im_path)));
        fprintf(fid,'%s %d',im_path{i},label{i});
        if i ~= numel(im_path)
            fprintf(fid,'\n');
        end
    end
    fclose(fid);
    % fprintf('\n');
    fprintf('done.\n');
end

if ~exist(lmdb_dir_tr,'dir')
    % generating lmdb
    %   cannot run this in matlab ...
    %   is this a bug?
    %   What ? I can run on flux?
    fprintf('creating train lmdb ... \n');
    cmd = sprintf('GLOG_logtostderr=1 %s/build/tools/convert_imageset --resize_height=%d --resize_width=%d %s %s %s', ...
        caffe_root, ...
        rsize(1), ...
        rsize(2), ...
        [im_root 'train2015/'], ...
        ip_file_tr, ...
        lmdb_dir_tr ...
        );
    system(cmd);
    % fprintf('%s\n',cmd);
    % fprintf('\n');
    fprintf('done.\n');
end

if ~exist(lmdb_dir_vl,'dir')
    fprintf('copying training lmdb to val lmdb ... \n');
    cmd = sprintf('cp -r %s %s',lmdb_dir_tr,lmdb_dir_vl);
    system(cmd);
    % fprintf('\n');
    fprintf('done.\n');
end

end


function [ train_imdata, train_labels, is_skip ] = load_imdata_label( list, list_train, anno_train, im_root, op_type, is_flip )
% load train_imdata & train_label for is_flip == 0 or is_flip == 1

switch op_type
    case 'single'
        num_class  = numel(list);
        num_im_cnt = sum(anno_train(:) == 1);
    case 'multiple'
        num_im_cnt = size(anno_train, 2);
        anno_train_t = anno_train;
        anno_train_t(anno_train_t == -1) = 0;
        anno_train_t(anno_train_t == -2) = 0;
        [lbl_uniq, ~, lbl_all] = unique(anno_train_t', 'rows');
        lbl_uniq = lbl_uniq';
        lbl_all  = lbl_all';
        num_class  = size(lbl_uniq, 2);
end

fprintf('num class: %d\n',num_class);

train_imdata = repmat(struct('im_path',[],'is_flip',[],'action',[]),[num_im_cnt 1]);
train_labels = repmat(struct('label',[]),[num_im_cnt 1]);

cnt     = 0;
is_skip = zeros(num_im_cnt,1);

for i = 1:num_class
    switch op_type
        case 'single'
            ii = find(anno_train(i,:) == 1);
        case 'multiple'
            ii = find(lbl_all == i);
    end
    for j = ii
        cnt = cnt + 1;
        
        % get feature files
        [~,fname,~] = fileparts(list_train{j});
        im_file     = [im_root 'train2015/' fname '.jpg'];  % hard-coded
        
        if ~exist(im_file,'file')
            is_skip(cnt) = 1;
            continue
        end
        train_imdata(cnt).im_path  = [fname '.jpg'];  % hard-coded
        train_imdata(cnt).is_flip  = is_flip;
        switch op_type
            case 'single'
                train_imdata(cnt).action = list(i);
            case 'multiple'
                train_imdata(cnt).action = list(find(lbl_all(:,i)));
        end
        
        % train_labels(cnt).label = i;
        train_labels(cnt).label = i-1; % label should start from 0
    end
end

assert(cnt == num_im_cnt);

end
