import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad, dur_to_mel2ph
from utils.pitch_tools import f0_to_coarse, denorm_f0, cwt2f0_norm

from .blocks import (
    Embedding,
    SinusoidalPositionalEmbedding,
    LayerNorm,
    LinearNorm,
    ConvNorm,
    BatchNorm1dTBC,
    EncSALayer,
    Mish,
    DiffusionEmbedding,
    ResidualBlock,
)
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm="ln", ffn_padding="SAME", ffn_act="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size,
            padding=ffn_padding,
            norm=norm, act=ffn_act)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)
    """这是一个Transformer编码器层的定义，它使用了一种名为EncSALayer的自定义层。让我解释一下它的各个部分：

- `hidden_size`: 表示隐藏层的大小，也是输入和输出的特征维度。
- `dropout`: 表示应用在多头自注意力和前馈神经网络(FFN)中的丢弃率。
- `kernel_size`: 表示自注意力机制中使用的卷积核大小。如果为None，则不使用卷积层。
- `num_heads`: 表示自注意力机制中的注意力头的数量。
- `norm`: 表示在注意力机制和前馈神经网络中使用的归一化方法。通常有"ln"（层归一化）和"bn"（批归一化）两种选择。
- `ffn_padding`: 表示前馈神经网络的填充方式，常见的有"SAME"和"VALID"两种选择。
- `ffn_act`: 表示前馈神经网络中使用的激活函数，常见的有"gelu"、"relu"等。

这个编码器层的`forward`方法接受输入张量`x`，并将其传递给`EncSALayer`进行处理，然后返回结果。`EncSALayer`是一个自定义的自注意力层，处理输入张量并执行自注意力机制和前馈神经网络的计算。
"""


class FFTBlocks(nn.Module):
    def __init__(self, hidden_size, num_layers, max_seq_len=2000, ffn_kernel_size=9, dropout=None, num_heads=2,
                 use_pos_embed=True, use_last_norm=True, norm="ln", ffn_padding="SAME", ffn_act="gelu", use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = max_seq_len
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=max_seq_len,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads, ffn_padding=ffn_padding, ffn_act=ffn_act)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == "ln":
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == "bn":
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None
            """这个`FFTBlocks`类是一个多层的Transformer编码器堆叠，每层由一个`TransformerEncoderLayer`组成。让我解释一下它的各个参数和部分：

- `hidden_size`: 表示隐藏层的大小，也是输入和输出的特征维度。
- `num_layers`: 表示堆叠的Transformer编码器层数。
- `max_seq_len`: 表示输入序列的最大长度。
- `ffn_kernel_size`: 表示自注意力机制中使用的卷积核大小。
- `dropout`: 表示应用在多头自注意力和前馈神经网络中的丢弃率。
- `num_heads`: 表示自注意力机制中的注意力头的数量。
- `use_pos_embed`: 表示是否使用位置编码。
- `use_last_norm`: 表示是否在最后一层使用层归一化。
- `norm`: 表示在最后一层使用的归一化方法，可以是"ln"（层归一化）或"bn"（批归一化）。
- `ffn_padding`: 表示前馈神经网络的填充方式，常见的有"SAME"和"VALID"两种选择。
- `ffn_act`: 表示前馈神经网络中使用的激活函数，常见的有"gelu"、"relu"等。
- `use_pos_embed_alpha`: 表示是否使用位置编码的缩放系数。

该类的`forward`方法接受输入张量并将其传递到Transformer编码器的每一层中进行处理。如果设置了位置编码，它将通过`SinusoidalPositionalEmbedding`添加位置编码。然后，输入通过一系列的Transformer编码器
层进行处理，最后一层进行了归一化。
"""

    def forward(self, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x


class FastspeechEncoder(FFTBlocks):
    def __init__(self, config):
        max_seq_len = config["max_seq_len"]
        hidden_size = config["transformer"]["encoder_hidden"]
        super().__init__(
            hidden_size,
            config["transformer"]["encoder_layer"],
            max_seq_len=max_seq_len * 2,
            ffn_kernel_size=config["transformer"]["ffn_kernel_size"],
            dropout=config["transformer"]["encoder_dropout"],
            num_heads=config["transformer"]["encoder_head"],
            use_pos_embed=False, # use_pos_embed_alpha for compatibility
            ffn_padding=config["transformer"]["ffn_padding"],
            ffn_act=config["transformer"]["ffn_act"],
        )
        self.padding_idx = 0
        self.embed_tokens = Embedding(
            len(symbols) + 1, hidden_size, self.padding_idx
        )
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(
            hidden_size, self.padding_idx, init_size=max_seq_len,
        )
        """这是一个基于FFTBlocks的Fastspeech编码器类。它使用FFTBlocks来构建Transformer编码器。以下是该类的主要特点：

- 初始化函数`__init__`接收一个`config`字典作为参数，其中包含有关模型配置的信息。
- `hidden_size`是编码器的隐藏状态大小。
- `config["transformer"]["encoder_layer"]`指定了编码器中Transformer编码器层的数量。
- `max_seq_len`是输入序列的最大长度。
- `config["transformer"]["ffn_kernel_size"]`是FeedForward网络中卷积层的内核大小。
- `config["transformer"]["encoder_dropout"]`是编码器中的dropout率。
- `config["transformer"]["encoder_head"]`是编码器中的注意力头数。
- `use_pos_embed=False`指示不使用位置编码（为了与其他模型的兼容性）。
- `ffn_padding=config["transformer"]["ffn_padding"]`指定FeedForward网络中卷积层的padding方式。
- `ffn_act=config["transformer"]["ffn_act"]`指定FeedForward网络中的激活函数类型。
- `self.embed_tokens`是一个Embedding层，用于将输入符号嵌入到连续的向量空间中。
- `self.embed_positions`是一个SinusoidalPositionalEmbedding层，用于提供位置编码。

该类继承了FFTBlocks，因此包含了FFTBlocks的所有功能，并针对Fastspeech编码器进行了一些特定的配置。
"""

    def forward(self, txt_tokens, encoder_padding_mask):
        """

        :param txt_tokens: [B, T]
        :param encoder_padding_mask: [B, T]
        :return: {
            "encoder_out": [T x B x C]
        }
        """
        x = self.forward_embedding(txt_tokens)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x

    def forward_embedding(self, txt_tokens):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        positions = self.embed_positions(txt_tokens)
        x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    """`FastspeechEncoder`类的`forward`方法负责将文本符号编码成文本嵌入，并将其传递给编码器进行处理。该方法的输入是文本符号和编码器的填充掩码，输出是编码器的输出。

- `forward_embedding`方法用于对文本符号进行嵌入处理。它首先使用`embed_tokens`将文本符号嵌入到连续的向量空间中，然后使用`embed_positions`生成位置编码，并将其与嵌入的文本符号相加。最后，应用
dropout以防止过拟合。

- `forward`方法首先调用`forward_embedding`对文本符号进行嵌入处理，然后将嵌入的文本符号传递给父类的`forward`方法，即`FFTBlocks`类的`forward`方法，以便对文本符号进行进一步的编码处理。最终，返
回编码器的输出结果。
"""


class FastspeechDecoder(FFTBlocks):
    def __init__(self, config):
        super().__init__(
            config["transformer"]["decoder_hidden"],
            config["transformer"]["decoder_layer"],
            max_seq_len=config["max_seq_len"] * 2,
            ffn_kernel_size=config["transformer"]["ffn_kernel_size"],
            dropout=config["transformer"]["decoder_dropout"],
            num_heads=config["transformer"]["decoder_head"],
            ffn_padding=config["transformer"]["ffn_padding"],
            ffn_act=config["transformer"]["ffn_act"],
        )
        """`FastspeechDecoder`类是基于`FFTBlocks`的解码器模型。它继承了`FFTBlocks`类，用于构建具有多层Transformer解码器结构的模型。

在初始化过程中，它接收一个配置字典`config`作为参数，并使用其中的参数来设置解码器模型的各种参数，包括隐藏层大小、层数、最大序列长度、FFN（Feed Forward Network）的卷积核大小、dropout率、注意力头数等。

通过继承`FFTBlocks`类，`FastspeechDecoder`类具备了多层Transformer解码器的结构和功能，包括前向传播方法`forward`，其中包含了层次化的Transformer解码器结构。
"""


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config, train_config):
        super(VarianceAdaptor, self).__init__()
        self.preprocess_config = preprocess_config

        self.use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
        self.use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
        self.predictor_grad = model_config["variance_predictor"]["predictor_grad"]

        self.hidden_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.predictor_layers = model_config["variance_predictor"]["predictor_layers"]
        self.dropout = model_config["variance_predictor"]["dropout"]
        self.ffn_padding = model_config["transformer"]["ffn_padding"]
        self.kernel = model_config["variance_predictor"]["predictor_kernel"]
        self.duration_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=self.filter_size,
            n_layers=model_config["variance_predictor"]["dur_predictor_layers"],
            dropout_rate=self.dropout, padding=self.ffn_padding,
            kernel_size=model_config["variance_predictor"]["dur_predictor_kernel"],
            dur_loss=train_config["loss"]["dur_loss"])
        self.length_regulator = LengthRegulator()
        if self.use_pitch_embed:
            n_bins = model_config["variance_embedding"]["pitch_n_bins"]
            self.pitch_type = preprocess_config["preprocessing"]["pitch"]["pitch_type"]
            self.use_uv = preprocess_config["preprocessing"]["pitch"]["use_uv"]

            if self.pitch_type == "cwt":
                self.cwt_std_scale = model_config["variance_predictor"]["cwt_std_scale"]
                h = model_config["variance_predictor"]["cwt_hidden_size"]
                cwt_out_dims = 10
                if self.use_uv:
                    cwt_out_dims = cwt_out_dims + 1
                self.cwt_predictor = nn.Sequential(
                    nn.Linear(self.hidden_size, h),
                    PitchPredictor(
                        h,
                        n_chans=self.filter_size,
                        n_layers=self.predictor_layers,
                        dropout_rate=self.dropout, odim=cwt_out_dims,
                        padding=self.ffn_padding, kernel_size=self.kernel))
                self.cwt_stats_layers = nn.Sequential(
                    nn.Linear(self.hidden_size, h), nn.ReLU(),
                    nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2)
                )
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size,
                    n_chans=self.filter_size,
                    n_layers=self.predictor_layers,
                    dropout_rate=self.dropout,
                    odim=2 if self.pitch_type == "frame" else 1,
                    padding=self.ffn_padding, kernel_size=self.kernel)
            self.pitch_embed = Embedding(n_bins, self.hidden_size, padding_idx=0)
        if self.use_energy_embed:
            self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
                "feature"
            ]
            assert self.energy_feature_level in ["phoneme_level", "frame_level"]
            energy_quantization = model_config["variance_embedding"]["energy_quantization"]
            assert energy_quantization in ["linear", "log"]
            n_bins = model_config["variance_embedding"]["energy_n_bins"]
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                energy_min, energy_max = stats["energy"][:2]

            self.energy_predictor = EnergyPredictor(
                self.hidden_size,
                n_chans=self.filter_size,
                n_layers=self.predictor_layers,
                dropout_rate=self.dropout, odim=1,
                padding=self.ffn_padding, kernel_size=self.kernel)
            if energy_quantization == "log":
                self.energy_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.energy_bins = nn.Parameter(
                    torch.linspace(energy_min, energy_max, n_bins - 1),
                    requires_grad=False,
                )
            self.energy_embedding = Embedding(n_bins, self.hidden_size, padding_idx=0)
            """`VarianceAdaptor`是一个神经网络模块，用于对FastSpeech模型中的语音特征进行方差调整。这种方差调整可以包括调整语音的持续时间、音高和能量等。

在该类中，主要包含以下功能：

1. **持续时间预测器（Duration Predictor）：** 该预测器用于预测目标音频帧的持续时间。在FastSpeech中，这对应于文本中每个音素（或帧）的持续时间。

2. **持续时间调整（Length Regulator）：** 用于根据持续时间预测结果，调整语音特征的持续时间。它将根据预测的持续时间对输入的语音特征进行扩展或缩放。

3. **音高预测器（Pitch Predictor）：** 如果配置了使用音高嵌入，则会使用该预测器预测目标音频帧的音高。音高预测器可以基于连续小波变换（CWT）或基于帧级别的音高。在FastSpeech中，音高通常用于控制
语音的音调。

4. **能量预测器（Energy Predictor）：** 如果配置了使用能量嵌入，则会使用该预测器预测目标音频帧的能量。能量通常用于控制语音的响度或音量。

这些预测器和调整器的配置和实现方式将根据FastSpeech模型的具体要求和架构而有所不同。
"""

    def get_pitch_embedding(self, decoder_inp, f0, uv, mel2ph, control, encoder_out=None):
        pitch_pred = f0_denorm = cwt = f0_mean = f0_std = None
        if self.pitch_type == "ph":
            pitch_pred_inp = encoder_out.detach() + self.predictor_grad * (encoder_out - encoder_out.detach())
            pitch_padding = encoder_out.sum().abs() == 0
            pitch_pred = self.pitch_predictor(pitch_pred_inp) * control
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
            f0_denorm = denorm_f0(f0, None, self.preprocess_config["preprocessing"]["pitch"], pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
            pitch = F.pad(pitch, [1, 0])
            pitch = torch.gather(pitch, 1, mel2ph)  # [B, T_mel]
            pitch_embed = self.pitch_embed(pitch)
        else:
            decoder_inp = decoder_inp.detach() + self.predictor_grad * (decoder_inp - decoder_inp.detach())
            pitch_padding = mel2ph == 0

            if self.pitch_type == "cwt":
                pitch_padding = None
                cwt = cwt_out = self.cwt_predictor(decoder_inp) * control
                stats_out = self.cwt_stats_layers(encoder_out[:, 0, :])  # [B, 2]
                mean = f0_mean = stats_out[:, 0]
                std = f0_std = stats_out[:, 1]
                cwt_spec = cwt_out[:, :, :10]
                if f0 is None:
                    std = std * self.cwt_std_scale
                    f0 = cwt2f0_norm(
                        cwt_spec, mean, std, mel2ph, self.preprocess_config["preprocessing"]["pitch"],
                    )
                    if self.use_uv:
                        assert cwt_out.shape[-1] == 11
                        uv = cwt_out[:, :, -1] > 0
            elif self.preprocess_config["preprocessing"]["pitch"]["pitch_ar"]:
                pitch_pred = self.pitch_predictor(decoder_inp, f0 if self.training else None) * control
                if f0 is None:
                    f0 = pitch_pred[:, :, 0]
            else:
                pitch_pred = self.pitch_predictor(decoder_inp) * control
                if f0 is None:
                    f0 = pitch_pred[:, :, 0]
                if self.use_uv and uv is None:
                    uv = pitch_pred[:, :, 1] > 0

            f0_denorm = denorm_f0(f0, uv, self.preprocess_config["preprocessing"]["pitch"], pitch_padding=pitch_padding)
            if pitch_padding is not None:
                f0[pitch_padding] = 0

            pitch = f0_to_coarse(f0_denorm)  # start from 0
            pitch_embed = self.pitch_embed(pitch)

        pitch_pred = {
            "pitch_pred": pitch_pred,
            "f0_denorm": f0_denorm,
            "cwt": cwt,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
        }

        return pitch_pred, pitch_embed
    """此代码段实现了`VarianceAdaptor`类中的`get_pitch_embedding`方法。该方法用于获取音高（pitch）的嵌入向量，并根据FastSpeech模型的配置和输入条件进行音高预测。

具体来说，该方法执行以下操作：

1. 根据`pitch_type`参数选择不同的音高预测策略，包括 `"ph"`（基于音素）、`"cwt"`（连续小波变换）和其他类型。
2. 对于基于音素的预测，将输入通过音高预测器进行预测，然后根据预测结果选择相应的音高序列，并获取对应的音高嵌入向量。
3. 对于其他类型的音高预测（如连续小波变换），根据预测结果进行进一步处理，并获取音高嵌入向量。
4. 返回预测的音高信息以及对应的音高嵌入向量。

该方法的实现涉及许多细节，例如处理不同类型的音高预测、数据填充等。这些细节可以根据FastSpeech模型的具体要求和预处理配置而有所不同。
"""

    def get_energy_embedding(self, x, target, mask, control):
        x.detach() + self.predictor_grad * (x - x.detach())
        prediction = self.energy_predictor(x, squeeze=True)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    """此代码段实现了 `VarianceAdaptor` 类中的 `get_energy_embedding` 方法，用于获取能量嵌入。

该方法接收输入特征 `x`、目标能量 `target`、掩码 `mask` 和能量控制参数 `control`。它执行以下操作：

1. 将输入特征 `x` 更新为其自身加上预测梯度的调整值，这样可以使梯度反向传播到输入特征中。
2. 使用能量预测器对输入特征进行预测，得到能量预测值。
3. 如果提供了目标能量，则根据目标能量将能量嵌入索引到相应的桶中。
4. 如果没有提供目标能量，则根据预测能量和控制参数将能量预测值调整到相应的范围，并将其索引到相应的桶中。
5. 返回能量预测值和能量嵌入。

这个方法主要用于获取输入特征的能量嵌入，以便在后续的模型计算中使用。
"""

    def forward(
        self,
        x,
        src_mask,
        max_src_len,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        mel2ph=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        speaker_emb=None,
    ):
        pitch_prediction = energy_prediction = None

        if speaker_emb is not None:
            x = x + speaker_emb.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        output_1 = x.clone()
        log_duration_prediction = self.duration_predictor(
            x.detach() + self.predictor_grad * (x - x.detach()), src_mask
        )
        if self.use_energy_embed and self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            output_1 += energy_embedding
        x = output_1.clone()

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            mel2ph = dur_to_mel2ph(duration_rounded, src_mask)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        output_2 = x.clone()
        if self.use_pitch_embed: # and self.pitch_type in ["frame", "cwt"]:
            if pitch_target is not None:
                if self.pitch_type == "cwt":
                    cwt_spec = pitch_target[f"cwt_spec"]
                    f0_mean = pitch_target["f0_mean"]
                    f0_std = pitch_target["f0_std"]
                    pitch_target["f0"] = cwt2f0_norm(
                        cwt_spec, f0_mean, f0_std, mel2ph, self.preprocess_config["preprocessing"]["pitch"],
                    )
                    pitch_target.update({"f0_cwt": pitch_target["f0"]})
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    x, pitch_target["f0"], pitch_target["uv"], mel2ph, p_control, encoder_out=output_1
                )
            else:
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    x, None, None, mel2ph, p_control, encoder_out=output_1
                )
            output_2 += pitch_embedding
        if self.use_energy_embed and self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            output_2 += energy_embedding
        x = output_2.clone()

        return (
            x,
            pitch_target,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )
    """此代码段实现了`VarianceAdaptor`类中的`forward`方法，用于在FastSpeech模型的推理过程中进行变异适应。

该方法接受一系列输入参数，包括输入特征 `x`、源序列掩码 `src_mask`、最大源长度 `max_src_len`、目标音高 `pitch_target`、目标能量 `energy_target`、目标持续时间 `duration_target`、mel2ph 映射、
音高控制参数 `p_control`、能量控制参数 `e_control`、持续时间控制参数 `d_control` 和说话人嵌入 `speaker_emb`。

该方法执行以下操作：

1. 将说话人嵌入添加到输入特征 `x` 中（如果提供了说话人嵌入）。
2. 获取预测的持续时间并根据掩码 `src_mask` 进行处理。
3. 如果使用能量嵌入并且能量特征级别为音素级别，则获取能量嵌入，并将其添加到输出中。
4. 根据目标持续时间或预测持续时间（如果没有目标持续时间）调整输入的长度，并相应地更新 `mel2ph` 和 `mel_mask`。
5. 如果使用音高嵌入，则获取音高嵌入，并将其添加到输出中。
6. 如果使用能量嵌入并且能量特征级别为帧级别，则获取能量嵌入，并将其添加到输出中。
7. 返回输出特征 `x`、音高目标、音高预测、能量预测、持续时间预测、四舍五入的持续时间、mel长度和mel掩码。

该方法的实现涉及到音高、能量和持续时间的预测，以及根据预测结果进行输入特征的调整和更新。
"""


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
    """`LengthRegulator`类是一个长度调节器，用于调整输入序列的长度以匹配目标长度。该类实现了`LR`方法和`forward`方法。

- `LR`方法接收输入张量`x`、预测的目标长度`duration`以及最大长度`max_len`。它首先遍历批次中的每个序列，并使用`expand`方法将每个序列扩展到对应的目标长度。然后，根据`max_len`进行填充处理，最终返回调节后的序列以及序列的长度。

- `expand`方法接收一个序列`batch`和对应的目标长度`predicted`，并将序列扩展到目标长度。具体操作是对每个向量进行复制以填充到目标长度，然后将扩展后的序列进行拼接。

- `forward`方法是`LengthRegulator`类的前向传播方法，它调用了`LR`方法对输入进行长度调节，并返回调节后的序列以及序列的长度。
"""


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The outputs are calculated in log domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding="SAME", dur_loss="mse"):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dur_loss = dur_loss
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        if self.dur_loss in ["mse", "huber"]:
            odims = 1
        elif self.dur_loss == "mog":
            odims = 15
        elif self.dur_loss == "crf":
            odims = 32
            from torchcrf import CRF
            self.crf = CRF(odims, batch_first=True)
        self.linear = torch.nn.Linear(n_chans, odims)

    def forward(self, xs, x_masks=None):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]

        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * (1 - x_masks.float())[:, :, None]  # (B, T, C)
        if self.dur_loss in ["mse"]:
            xs = xs.squeeze(-1)  # (B, Tmax)
        return xs
    """这段代码实现了持续时间预测器模块，用于预测每个帧的持续时间。这是FastSpeech模型的一部分，FastSpeech是一种用于文本到语音合成的模型。

- `__init__` 方法初始化了持续时间预测器模块，定义了卷积层、激活函数、归一化层和线性层等组件，并根据选择的损失函数选择了输出的维度。
- `forward` 方法实现了前向传播过程，首先将输入转置以匹配卷积层的输入要求，然后通过卷积层进行特征提取，接着通过线性层输出持续时间预测值，并根据输入的掩码进行适当的屏蔽处理。

这个模块的输出是持续时间的预测值，可以用于后续的模型训练和推理过程中。
"""


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding="SAME"):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs, squeeze=False):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs.squeeze(-1) if squeeze else xs
    """这段代码实现了一个音高预测器模块，用于从输入的特征中预测音高值。该模块的结构包括了卷积层、激活函数、归一化层、线性层和位置编码。

- `__init__` 方法初始化了音高预测器模块，定义了卷积层、激活函数、归一化层和线性层等组件。可以根据需要调整卷积层的数量、通道数、内核大小和dropout率等超参数。
- `forward` 方法实现了前向传播过程，首先通过位置编码增强输入特征，然后通过卷积层进行特征提取，最后通过线性层输出音高预测值。

该模块的输出是音高的预测值，可以用于语音合成中的音高控制。
"""


class EnergyPredictor(PitchPredictor):
    pass


class Denoiser(nn.Module):
    """ Conditional Diffusion Denoiser """

    def __init__(self, preprocess_config, model_config):
        super(Denoiser, self).__init__()
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_encoder = model_config["transformer"]["encoder_hidden"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        residual_layers = model_config["denoiser"]["residual_layers"]
        dropout = model_config["denoiser"]["denoiser_dropout"]
        multi_speaker = model_config["multi_speaker"]

        self.input_projection = nn.Sequential(
            ConvNorm(n_mel_channels, residual_channels, kernel_size=1),
            nn.ReLU()
        )
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, residual_channels)
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_encoder, residual_channels, dropout=dropout, multi_speaker=multi_speaker
                )
                for _ in range(residual_layers)
            ]
        )
        self.skip_projection = ConvNorm(
            residual_channels, residual_channels, kernel_size=1
        )
        self.output_projection = ConvNorm(
            residual_channels, n_mel_channels, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.conv.weight)

    def forward(self, mel, diffusion_step, conditioner, speaker_emb, mask=None):
        """

        :param mel: [B, 1, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :param speaker_emb: [B, M]
        :return:
        """
        x = mel[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, conditioner, diffusion_step, speaker_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]

        return x[:, None, :, :]
    """这段代码实现了一个条件扩散去噪器模块。该模块的结构包括输入投影层、扩散嵌入、MLP、残差层、跳跃投影层和输出投影层。

- `__init__` 方法初始化了条件扩散去噪器模块，定义了输入投影层、扩散嵌入、MLP、残差层、跳跃投影层和输出投影层等组件。可以根据需要调整残差层的数量、残差通道数、dropout率等超参数。
- `forward` 方法实现了前向传播过程，首先通过输入投影层对输入特征进行投影，然后通过残差层处理特征，并记录跳跃连接用于后续处理，最后通过输出投影层生成去噪后的音频特征。

这个模块的输出是去噪后的音频特征，可以用于语音合成中的音频去噪。
"""
