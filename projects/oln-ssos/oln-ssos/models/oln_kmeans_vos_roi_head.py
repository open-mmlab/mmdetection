import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn, Tensor
from typing import Tuple, List

from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptMultiConfig, ConfigType, InstanceList
from projects.OLN.oln import OLNRoIHead

import torch.nn.functional as F


@MODELS.register_module()
class OLNKMeansVOSRoIHead(OLNRoIHead):

    def __init__(self,
                 start_epoch=0,
                 logistic_regression_hidden_dim=512,
                 vos_samples_per_class=1000,
                 negative_sampling_size=10000,
                 bottomk_epsilon_dist=1,
                 ood_loss_weight=0.1,
                 pseudo_label_loss_weight=1.0,
                 k=5,
                 recalculate_pseudolabels_every_epoch=1,
                 k_means_minibatch=True,
                 repeat_ood_sampling=4,
                 pseudo_bbox_roi_extractor: OptMultiConfig = None,
                 *args,
                 **kwargs):
        """
        VOS BBox Head

        Args:
            vos_sample_per_class: queue size for each class to form the Gaussians
            start_epoch: starting epoch where VOS is going to be applied
            logistic_regression_hidden_dim: hidden dimension for the logistic regression layer (phi in Eq. 5)
            negative_sampling_size: number of samples from the multivariate Gaussian where the lowest k samples are
                considered negative (ood)
            bottomk_epsilon_dist: lowest k elements to use from `negative_sampling_size` samples form
                the multivariate Gaussian to be considered as negative
            ood_loss_weight: uncertainty loss weight
        """
        super(OLNKMeansVOSRoIHead, self).__init__(*args, **kwargs)
        self.vos_samples_per_class = vos_samples_per_class
        self.start_epoch = start_epoch
        self.bottomk_epsilon_dist = bottomk_epsilon_dist
        self.negative_sampling_size = negative_sampling_size
        self.ood_loss_weight = ood_loss_weight

        self.k = k
        self.recalculate_pseudolabels_every_epoch = recalculate_pseudolabels_every_epoch
        self.k_means_minibatch = k_means_minibatch
        self.repeat_ood_sampling = repeat_ood_sampling

        self.logistic_regression_layer = nn.Sequential(
            nn.Linear(1, logistic_regression_hidden_dim),
            nn.ReLU(),
            nn.Linear(logistic_regression_hidden_dim, 1)
        )

        self.epoch = 0

        self.data_dict = torch.zeros(self.k, self.vos_samples_per_class, 1024).cuda()
        self.number_dict = {}
        for i in range(self.k):
            self.number_dict[i] = 0

        # self.samples_for_covariance = 20 * 1024
        self.ft_minibatches = []

        self.pseudo_score = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.k)
        )

        for m in self.pseudo_score.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
        ft_size = pseudo_bbox_roi_extractor['roi_layer']['output_size'] ** 2 * pseudo_bbox_roi_extractor['out_channels']
        self.means = nn.Parameter((torch.zeros(k, ft_size)), requires_grad=False)
        self.cov = None
        self.kmeans = None

        self.loss_pseudo_cls = torch.nn.CrossEntropyLoss()

        self.post_epoch_features = []
        self.pseudo_label_loss_weight = pseudo_label_loss_weight
        self.bbox_head.num_classes = self.k

        self.pseudo_bbox_roi_extractor = MODELS.build(pseudo_bbox_roi_extractor)

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            # Not ideal, but changing the labels to pseudo labels
            batch_gt_instances_i = batch_gt_instances[i].clone()
            batch_gt_instances_i.labels = batch_gt_instances_i.pseudo_labels

            assign_result = self.bbox_assigner.assign(
                rpn_results,
                batch_gt_instances_i,
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances_i,
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:  # Sampling results has the pseudo_labels stores in the labels field
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

            if 'loss_ood' in bbox_results.keys():
                losses.update({'loss_ood': bbox_results['loss_ood'] * self.ood_loss_weight})
                losses.update({'loss_pseudo_class': bbox_results['loss_pseudo_class'] * self.pseudo_label_loss_weight})

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=None,
            bbox_pred=bbox_results['bbox_pred'],
            bbox_score=bbox_results['bbox_score'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])

        # VOS STARTS HERE
        oln_ssos_loss = self._ood_forward_train(bbox_results, bbox_loss_and_target['bbox_targets'], device=x[0].device)
        bbox_results.update(loss_ood=oln_ssos_loss['loss_ood'], loss_pseudo_class=oln_ssos_loss['loss_pseudo_class'])
        return bbox_results

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor):
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, bbox_score, fts = self.bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, shared_bbox_feats=fts,
            bbox_score=bbox_score)
        return bbox_results

    def _ood_forward_train(self, bbox_results, bbox_targets, device):
        selected_fg_samples = (bbox_targets[0] != self.k).nonzero().view(-1)
        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
        gt_classes_numpy = bbox_targets[0].cpu().numpy().astype(int)

        gt_box_features = []
        for index in indices_numpy:
            gt_box_features.append(bbox_results['shared_bbox_feats'][index].view(1, -1))
        gt_box_features = torch.cat(gt_box_features, dim=0)

        ood_reg_loss = torch.zeros(1).to(device)
        loss_pseudo_score = torch.zeros(1).cuda()
        if self.kmeans is not None:
            gt_pseudo_logits = self.pseudo_score(gt_box_features)
            gt_pseudo_labels = bbox_targets[0][selected_fg_samples]
            loss_pseudo_score = self.loss_pseudo_cls(gt_pseudo_logits, gt_pseudo_labels.long())

            sum_temp = 0
            for index in range(self.k):
                sum_temp += self.number_dict[index]
            queue_ready = sum_temp >= self.k * self.vos_samples_per_class
            if not queue_ready:
                for index in indices_numpy:
                    fts = bbox_results['shared_bbox_feats'][index].detach()
                    dict_key = gt_classes_numpy[index]
                    if self.number_dict[dict_key] < self.vos_samples_per_class:
                        self.data_dict[dict_key][self.number_dict[dict_key]] = fts
                        self.number_dict[dict_key] += 1
            else:
                for index in indices_numpy:
                    fts = bbox_results['shared_bbox_feats'][index].detach()
                    dict_key = gt_classes_numpy[index]
                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                          fts.view(1, -1)), 0)
                if self.epoch >= self.start_epoch:
                    for index in range(self.k):
                        if index == 0:
                            X = self.data_dict[index] - self.data_dict[index].mean(0)
                            mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                        else:
                            X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                            mean_embed_id = torch.cat((mean_embed_id,
                                                       self.data_dict[index].mean(0).view(1, -1)), 0)

                    # add the variance.
                    temp_precision = torch.mm(X.t(), X) / len(X)
                    # for stable training.
                    temp_precision += 0.0001 * torch.eye(self.bbox_head.fc_out_channels, device=device)
                    ood_samples = None
                    for index in range(self.k):
                        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                            mean_embed_id[index], covariance_matrix=temp_precision)
                        for _ in range(self.repeat_ood_sampling):
                            negative_samples = new_dis.rsample((self.negative_sampling_size,))
                            prob_density = new_dis.log_prob(negative_samples)

                            # keep the data in the low density area.
                            cur_samples, index_prob = torch.topk(- prob_density, self.bottomk_epsilon_dist)
                            if ood_samples is None:
                                ood_samples = negative_samples[index_prob]
                            else:
                                ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                        del new_dis
                        del negative_samples

                    energy_score_for_fg = torch.logsumexp(gt_pseudo_logits, 1)

                    # Now we need to get the class logits for the negative samples.
                    predictions_ood = self.pseudo_score(ood_samples)
                    energy_score_for_bg = torch.logsumexp(predictions_ood, 1)

                    input_for_loss = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    id_labels_size = energy_score_for_fg.shape[0]
                    labels_for_loss = torch.cat((torch.ones(id_labels_size).to(device),
                                                 torch.zeros(len(ood_samples)).to(device)), -1)

                    output = self.logistic_regression_layer(input_for_loss.view(-1, 1))
                    ood_reg_loss = F.binary_cross_entropy_with_logits(
                        output.view(-1), labels_for_loss)

        return dict(loss_ood=ood_reg_loss, loss_pseudo_class=loss_pseudo_score)

    def accumulate_pseudo_labels(self, fts, rois):
        bbox_feats = self.pseudo_bbox_roi_extractor(
            fts[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = bbox_feats.flatten(1)
        self.post_epoch_features.append(bbox_feats)

    def calculate_pseudo_labels(self):
        if self.means.sum().cpu().item() == 0:
            self.kmeans = MiniBatchKMeans(n_clusters=self.k, n_init=1, batch_size=1024)
        else:
            self.kmeans = MiniBatchKMeans(n_clusters=self.k, n_init=1, batch_size=1024,
                                          init=self.means.data.cpu())

        current_iter = 0
        dev = self.post_epoch_features[0].device
        data_to_fit = torch.zeros((1024, self.post_epoch_features[0].shape[1])).to(dev)
        last_index = 0
        while current_iter < len(self.post_epoch_features):
            iter_fts = self.post_epoch_features[current_iter]
            if iter_fts is None:
                current_iter += 1
                continue
            n_fts = iter_fts.shape[0]
            if last_index + n_fts < 1024:
                data_to_fit[last_index:(last_index + n_fts)] = iter_fts
                last_index += n_fts
                current_iter += 1
            else:
                fts_to_use = 1024 - last_index
                if fts_to_use > 0:
                    data_to_fit[last_index:] = iter_fts[:fts_to_use]
                self.kmeans.partial_fit(data_to_fit.cpu())
                last_index = n_fts - fts_to_use
                data_to_fit = torch.zeros((1024, self.post_epoch_features[0].shape[1])).to(dev)
                data_to_fit[:last_index] = iter_fts[fts_to_use:]
                current_iter += 1
        labels = []
        for iter_fts in self.post_epoch_features:
            if iter_fts is None:
                continue
            _labels = self.kmeans.predict(iter_fts.cpu())
            labels.append(torch.tensor(_labels).to(dev))
        labels = torch.cat(labels).cpu()
        self.means.data = torch.tensor(self.kmeans.cluster_centers_).to(dev)
        self.post_epoch_features = []
        # total_samples = sum(self.kmeans.counts_)
        # cw = total_samples / (self.k * self.kmeans.counts_)
        # self.loss_pseudo_cls = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=fts.device))
        return labels

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Test only det bboxes without augmentation."""
        # RPN score
        rpn_scores = torch.cat([res.scores for res in rpn_results_list], 0)

        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        bbox_scores = bbox_results['bbox_score']

        # OOD
        inter_feats = self.pseudo_score(bbox_results['shared_bbox_feats'])  # N x (K + 1)
        energy = torch.logsumexp(inter_feats, 1)
        ood_scores = self.logistic_regression_layer(energy.view(-1, 1))

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)
        ood_scores = ood_scores.split(num_proposals_per_img, 0)
        bbox_scores = bbox_scores.split(num_proposals_per_img, 0)
        rpn_scores = rpn_scores.split(num_proposals_per_img, 0)
        inter_feats = inter_feats.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_preds is not None:
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            bbox_preds=bbox_preds,
            ood_scores=ood_scores,
            bbox_scores=bbox_scores,
            rpn_scores=rpn_scores,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list
