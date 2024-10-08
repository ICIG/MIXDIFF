import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeaker import embedding


class PreDefinedEmbedder(nn.Module):
    """ Speaker Embedder Wrapper """

    def __init__(self, config):
        super(PreDefinedEmbedder, self).__init__()
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]
        self.embedder_type = config["preprocessing"]["speaker_embedder"]
        self.embedder_cuda = config["preprocessing"]["speaker_embedder_cuda"]
        self.embedder = self._get_speaker_embedder()

    def _get_speaker_embedder(self):
        embedder = None
        if self.embedder_type == "DeepSpeaker":
            embedder = embedding.build_model(
                "./deepspeaker/pretrained_models/ResCNN_triplet_training_checkpoint_265.h5"
            )
        else:
            raise NotImplementedError
        return embedder

    def forward(self, audio):
        if self.embedder_type == "DeepSpeaker":
            spker_embed = embedding.predict_embedding(
                self.embedder,
                audio,
                self.sampling_rate,
                self.win_length,
                self.embedder_cuda
            )

        return spker_embed
    """这个类是一个包装器，用于调用预定义的说话人嵌入器模型来提取说话人嵌入。它具有以下方法和属性：

- `__init__`: 初始化方法，设置了从配置中获取的参数，并调用 `_get_speaker_embedder` 方法来初始化说话人嵌入器模型。

- `_get_speaker_embedder`: 根据配置中指定的说话人嵌入器类型，加载预训练模型。

- `forward`: 前向传播方法，用于提取输入音频的说话人嵌入。根据配置中的说话人嵌入器类型，调用相应的预训练模型来提取说话人嵌入。

这个类的功能是根据给定的音频提取说话人的嵌入表示。
"""
