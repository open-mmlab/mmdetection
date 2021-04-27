function evaluation(norm_pred_list,gt_dir,setting_name,setting_class,legend_name)
load(gt_dir);
if ~exist(sprintf('./plot/baselines/Val/%s/%s',setting_class,legend_name),'dir')
    mkdir(sprintf('./plot/baselines/Val/%s/%s',setting_class,legend_name));
end
IoU_thresh = 0.5;
event_num = 61;
thresh_num = 1000;
org_pr_cruve = zeros(thresh_num,2);
count_face = 0;

for i = 1:event_num
    img_list = file_list{i};
    gt_bbx_list = face_bbx_list{i};
    pred_list = norm_pred_list{i};
    sub_gt_list = gt_list{i};
    img_pr_info_list = cell(length(img_list),1);
    
    fprintf('%s, current event %d\n',setting_name,i);
    for j = 1:length(img_list)
        gt_bbx = gt_bbx_list{j};
        pred_info = pred_list{j};
        keep_index = sub_gt_list{j};
        count_face = count_face + length(keep_index);
        
        if isempty(gt_bbx) || isempty(pred_info)
            continue;
        end
        ignore = zeros(size(gt_bbx,1),1);
        if ~isempty(keep_index)
            ignore(keep_index) = 1;
        end
        
        [pred_recall, proposal_list] = image_evaluation(pred_info, gt_bbx, ignore, IoU_thresh);
        
        img_pr_info = image_pr_info(thresh_num, pred_info, proposal_list, pred_recall);
        img_pr_info_list{j} = img_pr_info;   
    end
    for j = 1:length(img_list)
        img_pr_info = img_pr_info_list{j};
        if ~isempty(img_pr_info)
            org_pr_cruve(:,1) = org_pr_cruve(:,1) + img_pr_info(:,1);
            org_pr_cruve(:,2) = org_pr_cruve(:,2) + img_pr_info(:,2);
        end
    end
end
pr_cruve = dataset_pr_info(thresh_num, org_pr_cruve, count_face);
save(sprintf('./plot/baselines/Val/%s/%s/wider_pr_info_%s_%s.mat',setting_class,legend_name,legend_name,setting_name),'pr_cruve','legend_name','-v7.3');
end

function [pred_recall,proposal_list] = image_evaluation(pred_info, gt_bbx, ignore, IoU_thresh)
    pred_recall = zeros(size(pred_info,1),1);
    recall_list = zeros(size(gt_bbx,1),1);
    proposal_list = zeros(size(pred_info,1),1);
    proposal_list = proposal_list + 1;
    pred_info(:,3) = pred_info(:,1) + pred_info(:,3); 
    pred_info(:,4) = pred_info(:,2) + pred_info(:,4);
    gt_bbx(:,3) = gt_bbx(:,1) + gt_bbx(:,3);
    gt_bbx(:,4) = gt_bbx(:,2) + gt_bbx(:,4);
    for h = 1:size(pred_info,1)
        overlap_list = boxoverlap(gt_bbx, pred_info(h,1:4));
        [max_overlap, idx] = max(overlap_list);
        if max_overlap >= IoU_thresh
            if (ignore(idx) == 0)
                recall_list(idx) = -1;
                proposal_list(h) = -1;
            elseif (recall_list(idx)==0)
                recall_list(idx) = 1;
            end
        end
        r_keep_index = find(recall_list == 1);
        pred_recall(h) = length(r_keep_index);
    end
end

function img_pr_info = image_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
    img_pr_info = zeros(thresh_num,2);       
    for t = 1:thresh_num
        thresh = 1-t/thresh_num;
        r_index = find(pred_info(:,5)>=thresh,1,'last');
        if (isempty(r_index))
            img_pr_info(t,2) = 0;
            img_pr_info(t,1) = 0;
        else
            p_index = find(proposal_list(1:r_index) == 1);
            img_pr_info(t,1) = length(p_index);
            img_pr_info(t,2) = pred_recall(r_index);
        end
    end
end

function pr_cruve = dataset_pr_info(thresh_num, org_pr_cruve, count_face)
    pr_cruve = zeros(thresh_num,2);
    for i = 1:thresh_num
        pr_cruve(i,1) = org_pr_cruve(i,2)/org_pr_cruve(i,1);
        pr_cruve(i,2) = org_pr_cruve(i,2)/count_face;
    end
end
