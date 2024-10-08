import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer_fs2"]["betas"],
            eps=train_config["optimizer_fs2"]["eps"],
            weight_decay=train_config["optimizer_fs2"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer_fs2"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer_fs2"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer_fs2"]["anneal_rate"]
        self.current_step = current_step
        self.last_lr = self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        """这个 `ScheduledOptim` 类是一个简单的学习率调度器的包装器，用于管理优化器的学习率。

- `__init__` 方法初始化了优化器，使用了 Adam 优化器，并设置了一些超参数，如动量项的权重（`betas`）、epsilon 值（`eps`）、权重衰减（`weight_decay`）等。同时，也设置了学习率的 warm-up 步数
（`n_warmup_steps`）、退火步数（`anneal_steps`）和退火速率（`anneal_rate`）。初始学习率 `init_lr` 根据模型配置中的编码器隐藏层大小动态计算得到。
  
这个类的作用是根据当前训练步数动态调整学习率，实现了一种渐进式的学习率调度策略，其中学习率在 warm-up 阶段逐渐增加，然后在退火阶段逐渐降低。
"""

    def get_last_lr(self):
        return self.last_lr

    def step(self):
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        self.last_lr = lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    """这个类的其他方法包括：

- `get_last_lr`: 返回上一次更新后的学习率。
  
- `step`: 执行一步优化，包括更新学习率和执行优化器的一步更新。
  
- `zero_grad`: 将优化器中的梯度清零。

- `load_state_dict`: 从保存的状态字典中加载优化器的状态。

- `_get_lr_scale`: 根据当前训练步数计算学习率的缩放比例，其中包括一个 warm-up 阶段和一个退火阶段。

- `_update_learning_rate`: 更新学习率的具体实现，根据 `_get_lr_scale` 计算得到的缩放比例更新优化器中的参数组的学习率，并返回当前学习率。
"""
