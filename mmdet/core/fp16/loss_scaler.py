class DynamicLossScaler:
    """
    Reference to
    "https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/loss_scaler.py"
    """
    def __init__(self,
                 init_scale=512,
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(
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

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
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

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)
