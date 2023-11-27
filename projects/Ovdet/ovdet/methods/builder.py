from mmengine.registry import Registry

OVD = Registry('baron', )
QUEUE = Registry('queue', )


def build_ovd(cfg):
    """Build backbone."""
    return OVD.build(cfg)


def build_queue(cfg):
    """Build backbone."""
    return QUEUE.build(cfg)
