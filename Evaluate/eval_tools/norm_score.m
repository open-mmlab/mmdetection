function norm_pred_list = norm_score(org_pred_list)

event_num = 61;
norm_pred_list = cell(event_num,1);
max_score = realmin('single');
min_score = realmax('single');
parfor i = 1:event_num
    pred_list = org_pred_list{i};
    for j = 1:size(pred_list,1)
        if(isempty(pred_list{j}))
            continue;
        end
        score_list = pred_list{j}(:,5);
        max_score = max(max_score,max(score_list));
        min_score = min(min_score,min(score_list));
    end
end

parfor i = 1:event_num
    fprintf('Norm prediction: current event %d\n',i);
    pred_list = org_pred_list{i};
    for j = 1:size(pred_list,1)
        if(isempty(pred_list{j}))
            continue;
        end
        score_list = pred_list{j}(:,5);
        norm_score_list = (score_list - min_score)/(max_score - min_score);
        pred_list{j}(:,5) = norm_score_list;
    end
    norm_pred_list{i} = pred_list;
end
