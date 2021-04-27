function wider_plot(set_list,dir_ext,seting_class,dateset_class)

method_list = dir(dir_ext);
model_num = size(method_list,1) - 2;
model_name = cell(model_num,1);

for i = 3:size(method_list,1)
    model_name{i-2} = method_list(i).name;
end

for i = 1:size(set_list,1)
    propose = cell(model_num,1);
    recall = cell(model_num,1);
    name_list = cell(model_num,1);
    ap_list = zeros(model_num,1);
    for j = 1:model_num
        load(sprintf('%s/%s/wider_pr_info_%s_%s.mat',dir_ext, model_name{j}, model_name{j}, set_list{i}));
        propose{j} = pr_cruve(:,2);
        recall{j} = pr_cruve(:,1);
        ap = VOCap(propose{j},recall{j});
        ap_list(j) = ap;
        ap = num2str(ap);
        if length(ap) < 5
            name_list{j} = [legend_name '-' ap];
        else
            name_list{j} = [legend_name '-' ap(1:5)];
        end       
    end
    [~,index] = sort(ap_list,'descend');
    propose = propose(index);
    recall = recall(index);
    name_list = name_list(index);
    plot_pr(propose, recall, name_list, seting_class, set_list{i},dateset_class);
end
