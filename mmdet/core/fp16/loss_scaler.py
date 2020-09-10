
class LossScaler:
    """
    Class that manages loss scaling. Reference to
    "https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/loss_scaler.py"

    Describle:
        Indirectly, by supplying ``mode='dynamic'`` for dynamic loss scaling.
        It's important to understand how :class:`LossScaler` operates.
        Loss scaling is designed to combat the problem of underflowing 
        gradients encountered at long times when training fp16 networks.  
        Dynamic loss scaling begins by attempting a very high loss
        scale.  Ironically, this may result in OVERflowing gradients.  
        If overflowing gradients are
        encountered, :class:`FP16_Optimizer` then skips the update step for 
        this particular iteration/minibatch, and :class:`LossScaler` adjusts 
        the loss scale to a lower value.  
        If a certain number of iterations occur without overflowing gradients 
        detected,:class:`LossScaler` increases the loss scale once more.
        In this way :class:`LossScaler` attempts to "ride the edge" of always 
        using the highest loss scale possible without incurring overflow.
        Args:
            init_scale (float, optional, default=512):  
                Initial loss scale attempted by :class:`LossScaler.`
            scale_factor (float, optional, default=2.0):  
                Factor used when adjusting the loss scale. If an overflow is 
                encountered, the loss scale is readjusted to 
                loss scale/``scale_factor``.  If ``scale_window`` 
                consecutive iterations take place without an overflow, 
                the loss scale is readjusted to loss_scale*``scale_factor``. 
            mode (str, optional, default='dynamic'): If using dynamic 
                loss scaling, 'static' for use constant loss scaling.
            scale_window (int, optional, default=1000):  
                Number of consecutive iterations without an overflow to 
                wait before increasing the loss scale.
        """
    def __init__(self,
                 init_scale=512,
                 mode='dynamic',
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.mode = mode
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    # `True` indicating there is 'inf/nan' in params. 
    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        if self.mode is not 'dynamic':
            return False
        for p in params:
            if p.grad is not None and LossScaler._has_inf_or_nan(
                    p.grad.data):
                return True
        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') \
                    or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed,
    # loss_scale only update if self.mode is 'dynamic'.
    def update_scale(self, overflow):
        if self.mode is not 'dynamic':
            return
        if overflow:
            self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % \
                    self.scale_window == 0:
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

