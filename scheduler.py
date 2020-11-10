import torch.optim as optim

class BoundingExponentialLR(optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, initial_lr=0.01, min_lr=0.001, last_epoch=-1):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        super().__init__(optimizer=optimizer, gamma=gamma, last_epoch=last_epoch)

    def _compute_lr(self, base_lr):
        if base_lr * self.gamma <= self.min_lr:
            return self.min_lr
        else:
            return base_lr * self.gamma

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs

        return [self._compute_lr(group['lr']) for group in self.optimizer.param_groups]