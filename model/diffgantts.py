import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LinearNorm, ConvNorm, DiffusionEmbedding, Mish
from .modules import FastspeechEncoder, FastspeechDecoder, VarianceAdaptor
from .diffusion import GaussianDiffusion
from utils.tools import get_mask_from_lengths
from .loss import get_adversarial_losses_fn


class DiffGANTTS(nn.Module):
    """ DiffGAN-TTS """

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(DiffGANTTS, self).__init__()
        self.model = args.model
        self.model_config = model_config

        self.text_encoder = FastspeechEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        if self.model in ["aux", "shallow"]:
            self.decoder = FastspeechDecoder(model_config)
            self.mel_linear = nn.Linear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )
        self.diffusion = GaussianDiffusion(args, preprocess_config, model_config, train_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )
                """这是一个名为 `DiffGANTTS` 的模型类，用于实现 DiffGAN-TTS 模型。这个类的功能如下：

- **初始化函数 (`__init__`)：** 初始化模型的各个组件，包括文本编码器 (`text_encoder`)、变分适配器 (`variance_adaptor`)、解码器 (`decoder`)、线性层 (`mel_linear`) 和扩散模型 (`diffusion`)。
根据模型配置中的设置，选择是否使用多说话人的嵌入 (`speaker_emb`)。

- **前向传播函数 (`forward`)：** 接收输入参数，根据模型的结构和配置执行前向传播操作。首先，将输入文本编码成特征向量。然后，通过变分适配器对特征向量进行调整，以生成预测的语音特征。如果模型类型是 
"aux" 或 "shallow"，则使用解码器将预测的特征映射到语谱图空间，并通过线性层进行投影。最后，通过扩散模型对生成的语谱图进行优化，以获得最终的输出。

这个类的设计使得它能够灵活地适应不同的模型配置和任务要求，并且可以轻松地与其他模型组件进行集成和扩展。
"""

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        mel2phs=None,
        spker_embeds=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        """这个 `forward` 方法用于执行模型的前向传播操作，接收多个输入参数，包括说话者 ID (`speakers`)、文本序列 (`texts`)、源长度 (`src_lens`)、最大源长度 (`max_src_len`)、语谱图 (`mels`)、
        语谱图长度 (`mel_lens`)、最大语谱图长度 (`max_mel_len`)、音高目标 (`p_targets`)、能量目标 (`e_targets`)、持续时间目标 (`d_targets`)、mel2ph 映射 (`mel2phs`)、
        说话者嵌入 (`spker_embeds`) 以及音高、能量和持续时间的控制因子 (`p_control`、`e_control`、`d_control`)。

在执行前向传播时，首先根据源长度和最大源长度生成源序列的掩码 (`src_masks`)。如果提供了语谱图长度和最大语谱图长度，也会生成相应的语谱图掩码 (`mel_masks`)。接着，根据输入的各种目标和控制因子执行模型
的具体计算过程，生成最终的输出结果。
"""

        output = self.text_encoder(texts, src_masks)

        speaker_emb = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_emb = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_emb = self.speaker_emb(spker_embeds) # [B, H]

        (
            output,
            p_targets,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            max_src_len,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            mel2phs,
            p_control,
            e_control,
            d_control,
            speaker_emb,
        )
        """这部分代码的主要作用是将文本序列 (`texts`) 通过文本编码器 (`text_encoder`) 进行编码，并根据说话者信息生成相应的说话者嵌入 (`speaker_emb`)。如果模型支持多说话者并且选择了嵌入方式，则
        根据提供的说话者 ID 或嵌入向量生成相应的说话者嵌入。接着，将编码后的文本序列和说话者嵌入作为输入，通过方差适配器 (`variance_adaptor`) 进行进一步处理，得到音高、能量、持续时间等预测结果以
        及相应的语谱图长度和掩码。

最终，该部分代码的输出包括经过编码器和方差适配器处理后的特征 (`output`)、音高目标、音高预测、能量预测、持续时间预测、持续时间四舍五入后的结果、语谱图长度以及语谱图掩码。
"""

        if self.model == "naive":
            cond = output.clone()
            coarse_mels = None
            (
                output, # x_0_pred
                x_ts,
                x_t_prevs,
                x_t_prev_preds,
                diffusion_step,
            ) = self.diffusion(
                mels,
                output,
                speaker_emb,
                mel_masks,
            )
        elif self.model in ["aux", "shallow"]:
            x_ts = x_t_prevs = x_t_prev_preds = diffusion_step = None
            cond = output.clone()
            coarse_mels = self.decoder(output, mel_masks)
            coarse_mels = self.mel_linear(coarse_mels)
            if self.model == "aux":
                output = self.diffusion.diffuse_trace(coarse_mels, mel_masks)
            elif self.model == "shallow":
                (
                    output, # x_0_pred
                    x_ts,
                    x_t_prevs,
                    x_t_prev_preds,
                    diffusion_step,
                ) = self.diffusion(
                    mels,
                    self._detach(cond),
                    self._detach(speaker_emb),
                    self._detach(mel_masks),
                    self._detach(coarse_mels),
                )
        else:
            raise NotImplementedError
        """在这段代码中，根据模型类型（`self.model`），选择相应的处理方式：

- 如果模型类型为 "naive"，则进行简单的扩散操作。将编码后的特征作为条件 (`cond`)，并通过扩散模型 (`diffusion`) 对语谱图 (`mels`) 进行扩散操作，得到扩散后的结果 (`output`)，以及一系列中间结果
 (`x_ts`, `x_t_prevs`, `x_t_prev_preds`) 和扩散步骤信息 (`diffusion_step`)。

- 如果模型类型为 "aux" 或 "shallow"，则首先根据编码后的特征通过解码器 (`decoder`) 生成粗糙的语谱图 (`coarse_mels`)，然后根据模型类型选择不同的处理方式：
  - 如果模型类型为 "aux"，则将生成的粗糙语谱图作为输入，通过扩散追踪方法 (`diffusion.diffuse_trace`) 进行扩散操作，得到最终结果 (`output`)。
  - 如果模型类型为 "shallow"，则将编码后的特征、粗糙的语谱图等作为输入，通过扩散模型 (`diffusion`) 进行扩散操作，得到最终结果 (`output`)，以及一系列中间结果
    (`x_ts`, `x_t_prevs`, `x_t_prev_preds`) 和扩散步骤信息 (`diffusion_step`)。

如果模型类型不是 "naive"、"aux" 或 "shallow" 中的任何一种，则抛出未实现错误。
"""

        return cond,[
            output,
            (x_ts, x_t_prevs, x_t_prev_preds),
            self._detach(speaker_emb),
            diffusion_step,
            p_predictions, # cannot detach each value in dict but no problem since loss will not use it
            self._detach(e_predictions),
            log_d_predictions, # cannot detach each value in dict but no problem since loss will not use it
            self._detach(d_rounded),
            self._detach(src_masks),
            self._detach(mel_masks),
            self._detach(src_lens),
            self._detach(mel_lens),
            ], p_targets, self._detach(coarse_mels)

    def _detach(self, p):
        return p.detach() if p is not None and self.model == "shallow" else p
    """这段代码中，返回了模型的输出以及一些额外信息。具体返回的内容如下：

- `cond`: 编码后的特征，作为条件。
- `[...]`: 列表中包含了一系列的输出和中间结果：
  - `output`: 模型的最终输出。
  - `(x_ts, x_t_prevs, x_t_prev_preds)`: 扩散过程中的一些中间结果。
  - `self._detach(speaker_emb)`: 如果存在说话者嵌入，则返回它的副本，否则返回 `None`。
  - `diffusion_step`: 扩散步骤信息。
  - `p_predictions`: 采样概率的预测值。
  - `self._detach(e_predictions)`: 能量的预测值的副本。
  - `log_d_predictions`: 离散概率的预测值。
  - `self._detach(d_rounded)`: 离散概率的四舍五入值的副本。
  - `self._detach(src_masks)`: 源序列的掩码。
  - `self._detach(mel_masks)`: 语谱图的掩码。
  - `self._detach(src_lens)`: 源序列的长度。
  - `self._detach(mel_lens)`: 语谱图的长度。
- `p_targets`: 采样概率的目标值。
- `self._detach(coarse_mels)`: 粗糙语谱图的副本，如果不存在则返回 `None`。

`_detach` 方法用于检查参数是否为 `None` 并且模型类型为 "shallow"，如果满足条件则对参数进行 `detach` 操作，否则直接返回参数。
"""


class JCUDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, preprocess_config, model_config, train_config):
        super(JCUDiscriminator, self).__init__()

        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        n_layer = model_config["discriminator"]["n_layer"]
        n_uncond_layer = model_config["discriminator"]["n_uncond_layer"]
        n_cond_layer = model_config["discriminator"]["n_cond_layer"]
        n_channels = model_config["discriminator"]["n_channels"]
        kernel_sizes = model_config["discriminator"]["kernel_sizes"]
        strides = model_config["discriminator"]["strides"]
        self.multi_speaker = model_config["multi_speaker"]

        self.input_projection = LinearNorm(2 * n_mel_channels, 2 * n_mel_channels)
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, n_channels[n_layer-1]),
        )
        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(
                LinearNorm(residual_channels, n_channels[n_layer-1]),
            )
        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer)
            ]
        )
        """这是一个称为`JCUDiscriminator`的类，表示了一个JCU鉴别器模型。以下是此类的一些关键特征：

- **初始化方法 (`__init__`)：** 在初始化方法中，定义了鉴别器的结构。它接受三个配置参数：`preprocess_config`、`model_config`和`train_config`。这些配置参数包含了预处理、模型和训练的相关参数。
  
- **输入投影 (`input_projection`)：** 这是一个线性层，用于对输入进行投影变换。它将输入的 mel 频谱特征维度从 `2 * n_mel_channels` 转换为相同大小的输出维度。

- **扩散嵌入 (`diffusion_embedding`)：** 这是一个扩散步骤嵌入器，用于将输入的频谱特征转换为嵌入向量。

- **多层感知机 (`mlp`)：** 这是一个包含两个线性层和一个激活函数的多层感知机。它将扩散嵌入的输出转换为特征向量。

- **多说话人处理 (`spk_mlp`)：** 如果模型支持多说话人，会定义一个额外的多层感知机。否则，将其设置为 `None`。

- **卷积块 (`conv_block`)：** 这是一个卷积块的列表，每个卷积块包含一个卷积层。这些卷积层用于在不同的层次上提取特征。

这个类定义了 JCU 鉴别器的整体结构，包括了输入处理、特征提取和多说话人处理等组件。
"""
        self.uncond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_uncond_layer)
            ]
        )
        self.cond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_cond_layer)
            ]
        )
        self.apply(self.weights_init)
        """在 `JCUDiscriminator` 类中，上述代码片段定义了两个卷积块 `uncond_conv_block` 和 `cond_conv_block`，它们是模型的一部分，并用于从特征中提取信息。

- **无条件卷积块 (`uncond_conv_block`)：** 这是一个模块列表，其中包含了一系列卷积层。这些层用于处理特征，以提取无条件的信息，不考虑说话人身份。

- **有条件卷积块 (`cond_conv_block`)：** 这也是一个模块列表，包含了一系列卷积层。与无条件卷积块类似，这些层用于处理特征，但这次是有条件的，它们会考虑说话人身份。

这两个卷积块的结构相似，只是它们的作用略有不同。通过这些卷积块，模型可以从输入的特征中提取不同层次的信息，其中有些信息是与说话人无关的，而另一些则是与说话人有关的。这对于鉴别器的任务来说是至关重要的。
"""

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts, x_t_prevs, s, t):
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        t -- [B]
        """
        x = self.input_projection(
            torch.cat([x_t_prevs, x_ts], dim=-1)
        ).transpose(1, 2)
        diffusion_step = self.mlp(self.diffusion_embedding(t)).unsqueeze(-1)
        if self.multi_speaker:
            speaker_emb = self.spk_mlp(s).unsqueeze(-1)

        cond_feats = []
        uncond_feats = []
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)

        x_cond = (x + diffusion_step + speaker_emb) \
            if self.multi_speaker else (x + diffusion_step)
        x_uncond = x

        for layer in self.cond_conv_block:
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
        return cond_feats, uncond_feats
    """这是`JCUDiscriminator`类的`forward`方法。它接受一些输入张量，并将它们传递到一系列卷积层中以提取特征。

- **输入：**
    - `x_ts`：时间步 `t` 处的当前信息（上游模型的输出）。
    - `x_t_prevs`：时间步 `t-1` 处的信息。
    - `s`：说话者的嵌入表示。
    - `t`：当前时间步数。

- **特征提取：**
    - 首先，通过`input_projection`将`x_ts`和`x_t_prevs`在最后一个维度上连接起来，并通过线性投影转换为与卷积层期望的输入形状一致。
    - 然后，通过一系列卷积块`conv_block`处理特征。这些卷积块中的每一个都会将输入特征映射到更高层次的特征表示，并且输出结果会存储在`cond_feats`和`uncond_feats`列表中，以备后续使用。
    - 对于有条件的特征，会将其与扩散步骤的表示和说话者的表示相加（如果启用了多说话者支持）。
    - 最后，将处理后的有条件特征和无条件特征返回。

在整个过程中，`leaky_relu`被用作激活函数，以确保在训练过程中梯度不会消失。
"""
