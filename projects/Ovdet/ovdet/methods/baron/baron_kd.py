import torch
import torch.nn.functional as F
from mmcv.ops import roi_align
from mmengine.runner.amp import autocast
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi, bbox_xyxy_to_cxcywh
from ...utils.misc import multi_apply
from ..builder import OVD
from .baron_base import BaronBase
from .boxes_cache import BoxesCache
from .neighborhood_sampling import NeighborhoodSampling
from .utils import repeat_crops_and_get_att_mask


def process_sampling_result_per_image(sampling_result, device):
    # add region dropout
    spanned_boxes = sampling_result.spanned_boxes
    normed_boxes = sampling_result.normed_boxes
    box_ids = sampling_result.box_ids
    seq_ids = [list(map(box_ids2seq_id, box_ids_)) for box_ids_ in box_ids]
    seq_ids_per_image = []
    start_id = 0
    for seq_ids_ in seq_ids:
        seq_ids_per_image.extend([box_id + start_id for box_id in seq_ids_])
        start_id += (max(seq_ids_) + 1)
    sampling_result.set_field(
        name='seq_ids',
        value=seq_ids_per_image,
        field_type='metainfo',
        dtype=None)

    group_split = [
        len(grp) * grp[0].shape[0] for ori in normed_boxes for grp in ori
    ]
    origin_split = [
        sum([len(grp) * grp[0].shape[0] for grp in ori])
        for ori in normed_boxes
    ]
    perms_split = [
        perm.shape[0] for ori in normed_boxes for grp in ori for perm in grp
    ]

    seq_level_origin_split = [
        sum([len(grp) for grp in ori]) for ori in normed_boxes
    ]
    seq_level_group_split = [len(grp) for ori in normed_boxes for grp in ori]

    normed_boxes = torch.cat(
        [torch.cat(grp, dim=0) for ori in normed_boxes for grp in ori],
        dim=0).to(device)
    spanned_boxes = torch.cat(
        [torch.stack(ori, dim=0) for ori in spanned_boxes]).to(device)

    return normed_boxes, spanned_boxes, origin_split, group_split, \
        perms_split, seq_level_origin_split, seq_level_group_split


def box_ids2seq_id(box_ids):
    box_ids_copy = box_ids.copy()
    box_ids_sorted = sorted(box_ids_copy, reverse=True)
    box_ids_str = ''.join([str(box_id) for box_id in box_ids_sorted])

    return int(box_ids_str)


@OVD.register_module()
class BaronKD(BaronBase):

    def __init__(self,
                 bag_weight,
                 single_weight,
                 use_attn_mask,
                 bag_temp,
                 single_temp,
                 use_gt,
                 clip_data_preprocessor,
                 boxes_cache=None,
                 **kwargs):
        super(BaronKD, self).__init__(**kwargs)
        self.neighborhood_sampling = NeighborhoodSampling(**self.sampling_cfg)
        self.bag_temp = bag_temp  # 30.0
        self.single_temp = single_temp  # 50.0
        self.use_attn_mask = use_attn_mask
        self.bag_weight = bag_weight
        self.single_weight = single_weight
        self.use_gt = use_gt
        self.clip_data_preprocessor = MODELS.build(clip_data_preprocessor)
        if boxes_cache is not None:
            boxes_cache.update(
                num_proposals=self.sampling_cfg['topk'],
                nms_thr=self.sampling_cfg['nms_thr'],
                score_thr=self.sampling_cfg['objectness_thr'])
            self.boxes_cache = BoxesCache(**boxes_cache)
        else:
            self.boxes_cache = None

    def _sample_on_topk(self, topk_proposals):
        img_shape = topk_proposals.img_shape
        h, w = img_shape
        device = topk_proposals.scores.device
        image_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)

        if len(topk_proposals) == 0:
            topk_proposals = InstanceData(
                bboxes=image_box[None],
                scores=torch.tensor([1.0], dtype=device),
                metainfo=topk_proposals.metainfo.copy())

        nmsed_proposals = self.preprocess_proposals(
            topk_proposals, image_box[None],
            self.sampling_cfg['shape_ratio_thr'],
            self.sampling_cfg['area_ratio_thr'],
            self.sampling_cfg['objectness_thr'], self.sampling_cfg['nms_thr'])
        if self.boxes_cache is not None:
            nmsed_proposals = self.boxes_cache(nmsed_proposals)
        func = self.neighborhood_sampling.sample
        boxes = nmsed_proposals.bboxes.tolist()
        groups_per_proposal, normed_boxes, spanned_boxes, box_ids = \
            multi_apply(func, boxes,
                        [img_shape] * len(
                            nmsed_proposals))  # can be time-consuming
        new_boxes = torch.cat([
            perm for single_proposal in groups_per_proposal
            for single_group in single_proposal for perm in single_group
        ],
                              dim=0).to(device)
        metainfo = topk_proposals.metainfo.copy()
        metainfo.update(
            normed_boxes=normed_boxes,
            spanned_boxes=spanned_boxes,
            box_ids=box_ids)
        sampled_instances = InstanceData(bboxes=new_boxes, metainfo=metainfo)

        return sampled_instances

    def _sample_topk_proposals(self, proposals_per_image):
        num = min(len(proposals_per_image), self.sampling_cfg['topk'])
        _, topk_inds = proposals_per_image.scores.topk(num)

        return proposals_per_image[topk_inds]

    @staticmethod
    def _add_gt_boxes(proposals, gt_boxes):
        if len(gt_boxes) == 0:
            return proposals
        proposal_bboxes = proposals.bboxes
        proposal_scores = proposals.scores
        gt_scores = torch.ones_like(gt_boxes[:, 0])

        return InstanceData(
            bboxes=torch.cat([gt_boxes, proposal_bboxes]),
            scores=torch.cat([gt_scores, proposal_scores]),
            metainfo=proposals.metainfo)

    def sample(self, rpn_results, batch_data_sample, **kwargs):
        rpn_results.set_metainfo(batch_data_sample.metainfo)
        topk_proposals = self._sample_topk_proposals(rpn_results)
        if self.use_gt:
            topk_proposals = self._add_gt_boxes(
                topk_proposals, batch_data_sample.gt_instances.bboxes)
        sampling_result = self._sample_on_topk(topk_proposals)

        return sampling_result

    @torch.no_grad()
    def _bbox_clip_image(self, spanned_boxes, clip_images, seqs_split_by_group,
                         normed_boxes_split_by_perms, clip_model):
        # TODO: repeat and mask
        image_encoder = clip_model.image_encoder
        num_groups_per_image = [b.shape[0] for b in spanned_boxes]
        clip_input_size = image_encoder.input_resolution

        clip_images = self.clip_data_preprocessor({'inputs':
                                                   clip_images})['inputs']

        input_to_clip = roi_align(clip_images, bbox2roi(spanned_boxes),
                                  (clip_input_size, clip_input_size), 1.0, 2,
                                  'avg', True)
        input_to_clip = input_to_clip.split(num_groups_per_image, dim=0)
        repeated_crops, attn_masks = multi_apply(
            repeat_crops_and_get_att_mask,
            input_to_clip,
            seqs_split_by_group,
            normed_boxes_split_by_perms,
            num_heads=image_encoder.num_heads,
            grid_size=image_encoder.attn_resolution,
            use_attn_mask=self.use_attn_mask)

        repeated_crops = torch.cat(repeated_crops, dim=0)
        if attn_masks[0] is None:
            attn_masks = None
        else:
            attn_masks = torch.cat(attn_masks, dim=0)
        clip_img_features, clip_img_tokens = image_encoder.encode_image(
            repeated_crops,
            normalize=True,
            return_tokens=True,
            attn_masks=attn_masks)
        return clip_img_features, clip_img_tokens

    def get_losses(self, pseudo_words, sampling_results, clip_model, images,
                   *args, **kwargs):
        image_ids = [res.img_id for res in sampling_results]
        device = pseudo_words.device
        # Note: perms = seq
        (normed_boxes, spanned_boxes, origin_split, group_split,
         preds_split_by_perms, seqs_split_split_by_origin,
         seqs_split_by_group) = multi_apply(
             process_sampling_result_per_image,
             sampling_results,
             device=device)
        positions = bbox_xyxy_to_cxcywh(torch.cat(normed_boxes, dim=0))
        position_embeddings = self.positional_embed(positions)
        pseudo_words = pseudo_words + position_embeddings
        word_masks = self._drop_word(pseudo_words)
        start_id = 0
        seq_ids = []
        for res in sampling_results:
            seq_ids_ = res['seq_ids']
            for seq_id in seq_ids_:
                seq_ids.append(seq_id + start_id)
            start_id += (max(seq_ids_) + 1)
        seq_ids = torch.tensor(
            seq_ids, dtype=torch.float32).to(device)  # avoid overflow
        normed_boxes_split_by_perms = [
            normed_boxes_.split(preds_split_by_perms_, dim=0)
            for normed_boxes_, preds_split_by_perms_ in zip(
                normed_boxes, preds_split_by_perms)
        ]
        # torch.cat(normed_boxes).split(preds_split_by_perms, dim=0)
        preds_split_by_perms = [p for b in preds_split_by_perms for p in b]
        word_sequences = pseudo_words.split(preds_split_by_perms, dim=0)
        word_masks = word_masks.split(preds_split_by_perms, dim=0)
        word_sequences = [
            seq.flatten(0, 1)[wm.flatten(0, 1)]
            for seq, wm in zip(word_sequences, word_masks)
        ]
        context_length = max([seq.shape[0] for seq in word_sequences])
        with autocast():
            text_encoder = clip_model.text_encoder
            # TODO: get local image tokens
            pseudo_text, end_token_ids = text_encoder.prepare_pseudo_text(
                word_sequences,
                context_length=context_length + 2)  # add start and stop token
            clip_text_features, clip_word_tokens = \
                text_encoder.encode_pseudo_text(pseudo_text, end_token_ids,
                                                text_pe=True, normalize=True,
                                                return_word_tokens=True)
            clip_text_features = clip_text_features.float()
            clip_image_features, clip_image_tokens = self._bbox_clip_image(
                spanned_boxes, images, seqs_split_by_group,
                normed_boxes_split_by_perms, clip_model)
        global_clip_image_features = self.queues.get_queue(
            'clip_image_features')
        global_clip_text_features = self.queues.get_queue('clip_text_features')
        num_queries = clip_text_features.shape[0]
        assert clip_image_features.shape[0] == num_queries
        label_mask = seq_ids[None] == seq_ids[:, None]
        label_mask.fill_diagonal_(False)
        # mask same synced_img
        img_ids = [
            torch.tensor(sum(b) * [img_id])
            for b, img_id in zip(seqs_split_split_by_origin, image_ids)
        ]
        img_ids = torch.cat(img_ids).to(device)
        global_text_feature_img_ids = global_clip_text_features[..., -1]
        global_image_feature_img_ids = global_clip_image_features[..., -1]

        # text features as queries
        image_keys = torch.cat(
            [clip_image_features, global_clip_image_features[..., :-1]], dim=0)
        similarity_matrix_0 = self.bag_temp * clip_text_features @ image_keys.T
        similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
        if global_image_feature_img_ids.shape[0] > 0:
            img_id_mask_0 = img_ids[:,
                                    None] == global_image_feature_img_ids[None]
            assert similarity_matrix_0[:,
                   num_queries:].shape == img_id_mask_0.shape, \
                f'image_ids: {img_ids}, {image_ids}, ' \
                f'{len(seqs_split_split_by_origin)} '
            similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
        # image features as queries
        text_keys = torch.cat(
            [clip_text_features, global_clip_text_features[..., :-1]], dim=0)
        similarity_matrix_1 = self.bag_temp * clip_image_features @ text_keys.T
        similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')
        if global_text_feature_img_ids.shape[0] > 0:
            img_id_mask_1 = img_ids[:,
                                    None] == global_text_feature_img_ids[None]
            similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')

        label = torch.arange(num_queries).to(device)

        loss = 0.5 * F.cross_entropy(similarity_matrix_0, label) \
            + 0.5 * F.cross_entropy(similarity_matrix_1, label)
        losses = dict(loss_bag=loss * self.bag_weight)
        # Enqueue
        queues_update = dict(
            clip_text_features=torch.cat(
                [clip_text_features, img_ids.view(-1, 1)], dim=-1).detach(),
            clip_image_features=torch.cat(
                [clip_image_features, img_ids.view(-1, 1)], dim=-1).detach())

        if self.single_weight > 0.0:
            preds_split_by_batch = [n.shape[0] for n in normed_boxes]
            img_ids = [
                torch.tensor(b * [img_id])
                for b, img_id in zip(preds_split_by_batch, image_ids)
            ]
            img_ids = torch.cat(img_ids).to(device)
            normed_boxes = torch.cat(
                normed_boxes, dim=0).split(
                    preds_split_by_perms, dim=0)
            clip_patch_features = F.normalize(
                roi_align(clip_image_tokens,
                          bbox2roi(normed_boxes).to(clip_image_tokens.dtype),
                          (1, 1), float(clip_image_tokens.shape[-1]), 2, 'avg',
                          True)[..., 0, 0],
                dim=-1)
            num_words_per_pred = [wm.sum(-1).tolist() for wm in word_masks]
            clip_word_features = [
                tk.split(spl)
                for (tk, spl) in zip(clip_word_tokens, num_words_per_pred)
            ]
            clip_word_features = F.normalize(
                torch.stack([
                    feat.mean(0).float() for feats in clip_word_features
                    for feat in feats
                ],
                            dim=0),
                dim=-1)
            start_id = 0
            box_ids = []
            for res in sampling_results:
                for ori in res['box_ids']:
                    box_ids_per_ori = [
                        torch.tensor(perm, dtype=torch.float32) for perm in ori
                    ]  # avoid overflow
                    try:
                        box_ids_per_ori = torch.cat(box_ids_per_ori) + start_id
                    except RuntimeError:
                        from mmengine.logging import print_log
                        print_log(f'{box_ids_per_ori}, {start_id}')
                        exit()
                    start_id += (box_ids_per_ori.max().item() + 1)
                    box_ids.append(box_ids_per_ori)
            box_ids = torch.cat(box_ids).to(device)
            global_clip_word_features = self.queues.get_queue(
                'clip_word_features')
            global_clip_patch_features = self.queues.get_queue(
                'clip_patch_features')

            global_word_feature_img_ids = global_clip_word_features[..., -1]
            global_patch_feature_img_ids = global_clip_patch_features[..., -1]

            num_queries = clip_patch_features.shape[0]
            assert num_queries == clip_word_features.shape[0]

            # text features as queries
            image_keys = torch.cat(
                [clip_patch_features, global_clip_patch_features[..., :-1]])
            similarity_matrix_0 = \
                self.single_temp * clip_word_features @ image_keys.T
            if global_patch_feature_img_ids.shape[0] > 0:
                img_id_mask_0 = img_ids[:,
                                        None] == global_patch_feature_img_ids[
                                            None]
                similarity_matrix_0[:, num_queries:][img_id_mask_0] = float(
                    '-inf')
            # image features as queries
            text_keys = torch.cat(
                [clip_word_features, global_clip_word_features[..., :-1]])
            similarity_matrix_1 = \
                self.single_temp * clip_patch_features @ text_keys.T
            if global_word_feature_img_ids.shape[0] > 0:
                img_id_mask_1 = img_ids[:,
                                        None] == global_word_feature_img_ids[
                                            None]
                similarity_matrix_1[:, num_queries:][img_id_mask_1] = float(
                    '-inf')
            labels = torch.arange(num_queries, device=device)
            label_mask = box_ids[None] == box_ids[:, None]
            label_mask.fill_diagonal_(False)

            similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
            similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')

            loss = F.cross_entropy(similarity_matrix_0, labels) * 0.5 \
                + F.cross_entropy(similarity_matrix_1, labels) * 0.5
            losses.update(loss_single=loss * self.single_weight)

            queues_update.update(
                clip_word_features=torch.cat(
                    [clip_word_features,
                     img_ids.view(-1, 1)], dim=-1).detach(),
                clip_patch_features=torch.cat(
                    [clip_patch_features,
                     img_ids.view(-1, 1)], dim=-1).detach())
            self.queues.dequeue_and_enqueue(queues_update)

        return losses
