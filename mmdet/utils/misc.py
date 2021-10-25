import ast
import torch

from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.parallel import MMDistributedDataParallel
from mmdet.parallel import MMDataCPU


class ExtendedDictAction(DictAction):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            if '[' in val or '(' in val:
                val = ast.literal_eval(val)
            else:
                val = [self._parse_int_float_bool(v) for v in val.split(',')]
                if len(val) == 1:
                    val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def prepare_mmdet_model_for_execution(model, cfg, distributed=False):
    """
    Prepare model for execution.
    Return model MMDistributedDataParallel, MMDataParallel or MMDataCPU.

    :param model: Model.
    :param cfg: training mmdet config.
    :param distributed: Enable distributed training mode.
    :return:
    """
    if torch.cuda.is_available():
        if distributed:
            # put model on gpus
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDataCPU(model)
    return model
