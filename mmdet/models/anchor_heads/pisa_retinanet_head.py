from ..registry import HEADS
from .retina_head import RetinaHead


@HEADS.register_module
class PISARetinaHead(RetinaHead):

    pass
