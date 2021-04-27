function pred_list = read_pred(file_dir, gt_dir)

load(gt_dir);
event_num = 61;
pred_list = cell(event_num,1);

for i = 1:event_num
    fprintf('Read prediction: current event %d\n',i);
    img_list = file_list{i};
    img_num = size(img_list,1);
    bbx_list = cell(img_num,1);
    for j = 1:img_num
        if ~exist(sprintf('%s/%s/%s.txt',file_dir,event_list{i},img_list{j}),'file')
            fprintf('Can not find the prediction file %s %s \n',event_list{i},img_list{j});
            continue;
        end
        
        fid = fopen(sprintf('%s/%s/%s.txt',file_dir,event_list{i},img_list{j}),'r');
        tmp = textscan(fid,'%s','Delimiter','\n');
        tmp = tmp{1};
        fclose(fid);
        try
            bbx_num = tmp{2,1};
            bbx_num = str2num(bbx_num);
            bbx = zeros(bbx_num,5);
            if bbx_num ==0
                continue;
            end
            for k = 1:bbx_num
                raw_info = str2num(tmp{k+2,1});
                bbx(k,1) = raw_info(1);
                bbx(k,2) = raw_info(2);
                bbx(k,3) = raw_info(3);
                bbx(k,4) = raw_info(4);
                bbx(k,5) = raw_info(5);
            end
            [~, s_index] = sort(bbx(:,5),'descend');
            bbx_list{j} = bbx(s_index,:);
        catch
            fprintf('Invalid format %s %s\n',event_list{i},img_list{j});
        end
    end
    pred_list{i} = bbx_list;
end
