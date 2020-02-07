import pickle
import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.core.bbox import bbox2roi
class LoadImage(object):

  def __call__(self, results):
      if isinstance(results['img'], str):
          results['filename'] = results['img']
      else:
          results['filename'] = None
      img = mmcv.imread(results['img'])
      results['img'] = img
      results['img_shape'] = img.shape
      results['ori_shape'] = img.shape
      return results

def Resoning_module(model,img)  
'''
    input
        model: previous stage model, eg:faster rcnn
        img: eg:img = '/content/mmdetection/demo/demo.jpg'

    output:
        PEMrWg
'''  
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    img = data['img'][0]
    img_meta = data['img_meta'][0]

    #compute p
    x = model.extract_feat(img)

    # RPN forward
   

    rpn_outs = model.rpn_head(x)
    proposal_inputs = rpn_outs + (img_meta, model.test_cfg.rpn)
    proposal_list = model.rpn_head.get_bboxes(*proposal_inputs)

    if model.training = True:
    # assign gts and sample proposals
        
        bbox_assigner = build_assigner(model.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
            model.train_cfg.rcnn.sampler, context=model)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                gt_bboxes[i],
                                                gt_bboxes_ignore[i],
                                                gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
    #bbox head        
        rois = bbox2roi([res.bboxes for res in sampling_results])

    else:
        rois = bbox2roi(proposal_list)
        
    roi_feats = model.bbox_roi_extractor(
            x[:len(model.bbox_roi_extractor.featmap_strides)], rois)

    cls_score, bbox_pred = model.bbox_head(roi_feats)
    p = torch.nn.functional.softmax(cls_score, dim=1) 
    
    #Golab semantic pool M 
    parm={}
    for name,parameters in model.named_parameters():
        parm[name]=parameters
    M = parm['bbox_head.fc_cls.weight'][:,:]
    
    #graph E
    device = torch.cuda.current_device()
    cls_r_prob = pickle.load(open('Reasoning\VOC_graph_r.pkl', 'rb'))
    cls_r_prob = torch.from_numpy(cls_r_prob).to(device)

    cls_a_prob = pickle.load(open('Reasoning\VOC_graph_a.pkl', 'rb'))
    cls_a_prob = torch.from_numpy(cls_a_prob).to(device)

    cls_r_prob = cls_r_prob.type(torch.cuda.FloatTensor)
    cls_a_prob = cls_a_prob.type(torch.cuda.FloatTensor)
   
    #graph with global semantic pool
    EMr = torch.matmul(cls_r_prob, M)
    EMa = torch.matmul(cls_a_prob, M)

    PEMr = torch.matmul(p,EMr)
    PEMa = torch.matmul(p,EMa)
    Wg = torch.nn.Linear(1024, 256,bias=False).to(device)
    Enhanced_feature = Wg(PEMr)

    return Enhanced_feature
