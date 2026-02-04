from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super().step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a dict.

        This method ensures that only the state_dict of after_scheduler is saved,
        not the entire object (which would duplicate the optimizer state).
        """
        state = {
            'multiplier': self.multiplier,
            'total_epoch': self.total_epoch,
            'finished': self.finished,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
            '_step_count': self._step_count,
            '_get_lr_called_within_step': self._get_lr_called_within_step,
        }

        # Save _last_lr if it exists
        if hasattr(self, '_last_lr'):
            state['_last_lr'] = self._last_lr

        # Save only the state_dict of after_scheduler, not the object itself
        if self.after_scheduler is not None:
            state['after_scheduler_state'] = self.after_scheduler.state_dict()
            state['after_scheduler_type'] = type(self.after_scheduler).__name__

        return state

    def load_state_dict(self, state_dict):
        """Loads the scheduler state.

        This method restores the scheduler state from a saved state_dict.
        Supports both new format (with after_scheduler_state) and old format
        (with after_scheduler object) for backward compatibility.
        """
        self.multiplier = state_dict['multiplier']
        self.total_epoch = state_dict['total_epoch']
        self.finished = state_dict['finished']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self._step_count = state_dict['_step_count']
        self._get_lr_called_within_step = state_dict['_get_lr_called_within_step']

        # Restore _last_lr if it exists in the state_dict
        if '_last_lr' in state_dict:
            self._last_lr = state_dict['_last_lr']

        # Restore after_scheduler state (new format)
        if 'after_scheduler_state' in state_dict and self.after_scheduler is not None:
            self.after_scheduler.load_state_dict(state_dict['after_scheduler_state'])
        # Backward compatibility: handle old format where entire object was saved
        elif 'after_scheduler' in state_dict and self.after_scheduler is not None:
            old_scheduler = state_dict['after_scheduler']
            # If old format contains a scheduler object with state_dict method
            if hasattr(old_scheduler, 'state_dict'):
                try:
                    self.after_scheduler.load_state_dict(old_scheduler.state_dict())
                    print("Warning: Loaded scheduler from old format (with full object). "
                          "Future saves will use the new compact format.")
                except Exception as e:
                    print(f"Warning: Could not load after_scheduler state from old format: {e}")
                    print("Continuing with newly initialized after_scheduler.")
