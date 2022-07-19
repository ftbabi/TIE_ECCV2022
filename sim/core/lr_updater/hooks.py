from torch._six import inf
from mmcv.runner.hooks import HOOKS, Hook, LrUpdaterHook


@HOOKS.register_module()
class PlateauLrUpdaterHook(LrUpdaterHook):

    def __init__(self, 
        mode='min', factor=0.1, 
        patience=10, threshold=0.0001, 
        threshold_mode='rel', cooldown=0, 
        min_lr=0, eps=1e-08, metric_func=lambda x: x['loss'], **kwargs):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.min_lr = min_lr
        self.eps = eps
        # self.last_epoch = 0
        self.metric_func = metric_func
        self.last_lr = None

        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

        super(PlateauLrUpdaterHook, self).__init__(**kwargs)
        
    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
    
    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
    
    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold
    
    def _reduce_lr(self):
        old_lr = self.last_lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr <= self.eps:
            new_lr = old_lr
        return new_lr

    def get_lr(self, runner, base_lr):
        if self.last_lr is None:
            self.last_lr = base_lr
            if runner.epoch > self.patience:
                self.last_lr = runner.optimizer.param_groups[-1]['lr']
        else:
            metrics = self.metric_func(runner.outputs)
            current = float(metrics)
            # if epoch is None:
            #     epoch = self.last_epoch + 1
            # else:
            #     warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
            # self.last_epoch = epoch

            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self.last_lr = self._reduce_lr()
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        return self.last_lr