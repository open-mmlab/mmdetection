from ..builder import PIPELINES
from .compose import Compose
from . import MultiScaleFlipAug


@PIPELINES.register_module()
class CroppedTilesFlipAug(MultiScaleFlipAug):

    def __init__(self,
                 transforms,
                 tile_shape,  # sub image size by cropped
                 tile_overlap,
                 tile_scale=None,  # sub image resize to as input for net
                 scale_factor=None,  # sub image resize factor to as input for net
                 flip=False,
                 flip_direction='horizontal'):
        # force Collect pipeline step to collect key 'tile_offset'
        for transform in transforms:
            if transform['type'] == 'Collect':
                if 'meta_keys' not in transform:
                    transform['meta_keys'] = ('filename', 'ori_filename', 'ori_shape',
                                              'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                              'flip_direction', 'img_norm_cfg')
                transform['meta_keys'] += ('tile_offset',)

        assert isinstance(tile_shape, tuple)
        assert isinstance(tile_overlap, tuple)
        self.tile_shape = tile_shape
        self.tile_overlap = tile_overlap

        super(CroppedTilesFlipAug, self).__init__(transforms=transforms,
                                                  img_scale=tile_scale,
                                                  scale_factor=scale_factor,
                                                  flip=flip,
                                                  flip_direction=flip_direction)

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        h, w, c = results['img_shape']
        w_ovr, h_ovr = self.tile_overlap
        w_s, h_s = self.tile_shape
        for h_off in range(0, max(1, h - h_ovr), h_s - h_ovr):
            if h_off > 0:
                h_off = min(h - h_s, h_off)  # h_off + hs <= h if h_off > 0
            for w_off in range(0, max(1, w - w_ovr), w_s - w_ovr):
                if w_off > 0:
                    w_off = min(w - w_s, w_off)  # w_off + ws <= w if w_off > 0
                for scale in self.img_scale:
                    for flip in flip_aug:
                        for direction in self.flip_direction:
                            _results = results.copy()
                            _results['img'] = _results['img'][h_off:h_off + h_s, w_off:w_off + w_s, :]
                            _results['tile_offset'] = (w_off, h_off)  # (x, y)
                            _results['img_shape'] = (h_s, w_s, c)
                            # _results['scale'] = scale
                            _results[self.scale_key] = scale
                            _results['flip'] = flip
                            _results['flip_direction'] = direction
                            data = self.transforms(_results)
                            aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'tile_scale={self.img_scale}, '
        repr_str += f'tile_shape={self.tile_shape}, '
        repr_str += f'tile_overlap={self.tile_overlap}, '
        repr_str += f'flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str


@PIPELINES.register_module()
class NoAug(object):

    def __init__(self, transforms):
        self.transforms = Compose(transforms)

    def __call__(self, results):
        data = self.transforms(results)
        for key in data.keys():
            data[key] = [data[key]]
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms})'
        return repr_str