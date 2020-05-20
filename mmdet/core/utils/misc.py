from functools import partial

import mmcv
import numpy as np
import torch
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def arange(start=0,
           end=None,
           step=1,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False):
    if torch.onnx.is_in_onnx_export():
        if end is None:
            raise ValueError('End of range must be defined.')
        assert out is None
        assert layout == torch.strided

        start_tensor = torch.as_tensor(start, dtype=torch.long, device=device)
        end_tensor = torch.as_tensor(end, dtype=torch.long, device=device)
        n = end_tensor - start_tensor

        if isinstance(step, torch.Tensor):
            n = (n + step - 1) // step

        result = torch.ones(n, layout=layout, device=device)\
            .nonzero().view(-1) + start_tensor

        if isinstance(step, torch.Tensor) or step > 1:
            result = result.view(-1, step)\
                .index_select(1, torch.zeros(1, dtype=torch.long))\
                .view(-1)

        if dtype is not None:
            result = result.to(dtype)

        return result
    else:
        return torch.arange(
            start=start,
            end=end,
            step=step,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad)


def topk(x, k, dim=None, **kwargs):
    from torch.onnx import operators, is_in_onnx_export

    if dim is None:
        dim = x.dim() - 1

    if is_in_onnx_export():
        n = operators.shape_as_tensor(x)[dim].unsqueeze(0)
        if not isinstance(k, torch.Tensor):
            k = torch.tensor([k], dtype=torch.long)
        # Workaround for ONNXRuntime: convert values to int to get minimum.
        n = torch.min(torch.cat((k, n), dim=0).int()).long()
        # ONNX OpSet 10 does not support non-floating point input for TopK.
        original_dtype = x.dtype
        require_cast = original_dtype not in {
            torch.float16, torch.float32, torch.float64
        }
        if require_cast:
            x = x.to(torch.float32)
        values, keep = torch.topk(x, n, dim=dim, **kwargs)
        if require_cast:
            values = values.to(original_dtype)
    else:
        values, keep = torch.topk(
            x, min(int(k), x.shape[dim]), dim=dim, **kwargs)
    return values, keep


def meshgrid(x, y, row_major=True, flatten=True):
    n, m = y.shape[0], x.shape[0]
    yy = y.view(-1, 1).expand(n, m)
    xx = x.view(1, -1).expand(n, m)
    if flatten:
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def dummy_pad(x, padding):

    class DummyPad(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, padding):
            return torch.nn.functional.pad(x, padding)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = None
            if ctx.needs_input_grad[0]:
                grad_output = grad_input
            return grad_output, None

        @staticmethod
        def symbolic(g, x, padding):
            return g.op("Identity", x)

    return DummyPad.apply(x, padding)


def to_numpy(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    assert isinstance(x, np.ndarray)
    x = x.astype(dtype)
    return x
