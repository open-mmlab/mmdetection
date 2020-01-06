import torch

class AnchorGenerator_FB(object):

    def __init__(self,input_size, box_spec):        
        self.input_size = input_size
        self.box_spec = box_spec
        self.base_anchors = self.gen_base_anchors()


    @property
    def num_base_anchors(self):        
        return self.base_anchors.size(0)


    def gen_base_anchors(self):
        ctr = ((10000. ) / 2, (10000. ) / 2)         
        base_anchors = self._gen_base_anchors(self.box_spec,ctr)
        return base_anchors


    def _gen_base_anchors(self,box_spec,ctr):
        _box_spec = box_spec
        x_ctr, y_ctr = ctr
        base_anchors = list()
        for scale, _, n in _box_spec:
            w = scale
            h = scale
            center_ids = torch.arange(0,n).float()
            if scale == 32:
                new_centers_y = y_ctr*(1/n)*center_ids 
                new_centers_x = x_ctr*(1/n)*center_ids
            elif scale == 64:
                new_centers_y = y_ctr*(1/n)*center_ids 
                new_centers_x = x_ctr*(1/n)*center_ids
            else:
                new_centers_y = torch.Tensor([0.5*y_ctr]) 
                new_centers_x = torch.Tensor([0.5*x_ctr])
            new_centers_y = new_centers_y.unsqueeze(0).repeat(n,1)
            new_centers_x = new_centers_x.unsqueeze(1).repeat(1,n)
            centers = torch.stack([new_centers_y, new_centers_x], dim=2)
            sizes = torch.Tensor([h, w])
            sizes = sizes.unsqueeze(0).unsqueeze(0).repeat(n,n,1)
            _base_anchors = torch.stack([centers, sizes], dim=2).view(-1, 4)
            _cy=torch.index_select(_base_anchors, 1, torch.LongTensor([0]))
            _cx=torch.index_select(_base_anchors, 1, torch.LongTensor([1]))
            _h=torch.index_select(_base_anchors, 1, torch.LongTensor([2]))
            _w=torch.index_select(_base_anchors, 1, torch.LongTensor([3]))
            ymin, xmin = _cy - 0.5*(_h), _cx - 0.5*(_w)
            ymax, xmax = _cy + 0.5*(_h), _cx + 0.5*(_w)
            _base_anchors = torch.cat([xmin, ymin ,xmax ,ymax], dim=1)
            base_anchors.append(_base_anchors)
        base_anchors = torch.cat(base_anchors,0).view(-1, 4)
        return base_anchors


    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, device='cuda'):
        all_anchors = []
        _box_spec = self.box_spec
        feat_h, feat_w = featmap_size       
        ctr = (self.input_size/(feat_w),self.input_size/(feat_h))
        all_anchors = self._grid_anchors(featmap_size, ctr, _box_spec).round()
        return all_anchors


    def _grid_anchors(self, featmap_size, ctr, box_spec, device='cuda'):
        _base_anchors = self._gen_base_anchors(box_spec,ctr).to(device)
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * (self.input_size/feat_w)     
        shift_y = torch.arange(0, feat_h, device=device) * (self.input_size/feat_h)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(_base_anchors)
        _all_anchors = _base_anchors[None, :, :] + shifts[:, None, :]        
        _all_anchors = _all_anchors.view(-1, 4)
        return _all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
