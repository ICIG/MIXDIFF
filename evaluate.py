import argparse
import os
import json

import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tools import to_device, log, synth_one_sample
from model import DiffGANTTSLoss
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args, model, discriminator, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs


    # Get dataset
    dataset = Dataset(
        "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    """这段代码是用于评估模型在验证集上的性能。它首先创建了一个用于加载验证数据的数据集对象，然后使用该数据集创建了一个数据加载器 DataLoader。在创建 DataLoader 时，它使用了之前定义的数据集
    对象的 collate_fn 函数来对数据进行整理。

这个函数的输入参数包括：
- args: 命令行参数，其中包含有关数据集、预处理配置等的信息。
- model: 要评估的模型。
- discriminator: 判别器模型（如果适用）。
- step: 当前训练步骤。
- configs: 包含预处理、模型和训练配置的元组。
- logger: 日志记录器，用于记录评估结果。
- vocoder: 语音合成器（如果适用）。
- losses: 用于记录损失函数值的字典。
"""

    # Get loss function
    Loss = DiffGANTTSLoss(args, preprocess_config, model_config, train_config).to(device)

    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            """这部分代码用于获取损失函数对象并计算在验证集上的损失。首先，它创建了一个损失函数对象 Loss，并将其移到适当的设备上。然后，它初始化了一个列表 loss_sums，用于存储每个损失类型的总和。
            如果损失是一个字典，则为每个键初始化一个值为0的计数器；否则，直接初始化为0。

接下来，它使用 DataLoader 加载验证集数据，并迭代遍历每个 batch。对于每个 batch，它将其移动到指定的设备上，并计算损失。
"""

            with torch.no_grad():
                if args.model == "aux":

                    # Forward
                    output, p_targets, coarse_mels = model(*(batch[2:]))
                    # Update Batch
                    batch[9] = p_targets

                    (
                        fm_loss,
                        recon_loss,
                        mel_loss,
                        pitch_loss,
                        energy_loss,
                        duration_loss,
                    ) = Loss(
                        model,
                        batch,
                        output,
                    )
                    output[0] = output[0][0] # only x_0 is needed after calculating loss

                    G_loss = recon_loss
                    D_loss = fm_loss = adv_loss = torch.zeros(1).to(device)
                    """在这段代码中，使用 `torch.no_grad()` 上下文管理器来确保在前向推断过程中不计算梯度。然后，根据 `args.model` 的值选择执行不同的逻辑。针对 `aux` 模型：

1. 进行前向传播，得到模型的输出 `output`、声学特征的目标值 `p_targets` 和粗糙的梅尔频谱 `coarse_mels`。
2. 更新批处理数据中的声学特征目标值。
3. 调用损失函数 `Loss`，计算不同类型的损失，包括 `fm_loss`（频率掩码损失）、`recon_loss`（重构损失）、`mel_loss`（梅尔频谱损失）、`pitch_loss`（音高损失）、`energy_loss`（能量损失）
和 `duration_loss`（持续时间损失）。
4. 由于只需要计算重构损失作为生成器的损失，因此将 `G_loss` 设置为 `recon_loss`，而其他损失（包括鉴别器损失 `D_loss`、频率掩码损失 `fm_loss` 和对抗损失 `adv_loss`）均设置为零张量。
"""

                else: # args.model in ["naive", "shallow"]

                    #######################
                    # Evaluate Discriminator #
                    #######################

                    # Forward
                    cond,output, *_ = model(*(batch[2:]))

                    xs, spk_emb, t, mel_masks = *(output[1:4]), output[9]
                    x_ts, x_t_prevs, x_t_prev_preds, spk_emb, t = \
                        [x.detach() if x is not None else x for x in (list(xs) + [spk_emb, t])]

                    D_real_cond, D_real_uncond = discriminator(x_ts, x_t_prevs, spk_emb, t)
                    D_fake_cond, D_fake_uncond = discriminator(x_ts, x_t_prev_preds, spk_emb, t)

                    D_loss_real, D_loss_fake = Loss.d_loss_fn(D_real_cond[-1], D_real_uncond[-1], D_fake_cond[-1], D_fake_uncond[-1])

                    D_loss = D_loss_real + D_loss_fake
                    """在这段代码中，处理了 `naive` 和 `shallow` 模型的情况。

1. 该部分是评估鉴别器的逻辑。
2. 通过模型进行前向传播，获取模型的输出 `output` 和条件信息 `cond`。
3. 从模型输出中提取需要的信息，如音频特征 `x_ts`、先前的音频特征 `x_t_prevs`、先前的音频特征预测 `x_t_prev_preds`、说话人嵌入 `spk_emb` 和时间步 `t`。
4. 对提取的信息进行处理，将需要计算梯度的部分 `detach()`，以避免影响鉴别器的训练。
5. 使用鉴别器对真实音频和生成音频进行前向传播，得到条件真假音频的鉴别结果 `D_real_cond`, `D_real_uncond`, `D_fake_cond`, `D_fake_uncond`。
6. 调用损失函数 `Loss.d_loss_fn()`，计算真假音频的鉴别器损失 `D_loss_real` 和 `D_loss_fake`。
7. 将真假音频的鉴别器损失相加，得到总的鉴别器损失 `D_loss`。
"""

                    #######################
                    # Evaluate Generator #
                    #######################

                    # Forward
                    cond,output, p_targets, coarse_mels = model(*(batch[2:]))
                    # Update Batch
                    batch[9] = p_targets

                    (x_ts, x_t_prevs, x_t_prev_preds), spk_emb, t, mel_masks = *(output[1:4]), output[9]

                    D_fake_cond, D_fake_uncond = discriminator(x_ts, x_t_prev_preds, spk_emb, t)
                    D_real_cond, D_real_uncond = discriminator(x_ts, x_t_prevs, spk_emb, t)

                    adv_loss = Loss.g_loss_fn(D_fake_cond[-1], D_fake_uncond[-1])

                    (
                        fm_loss,
                        recon_loss,
                        mel_loss,
                        pitch_loss,
                        energy_loss,
                        duration_loss,
                    ) = Loss(
                        model,
                        batch,
                        output,
                        coarse_mels,
                        (D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond),
                    )
                    """在这段代码中，处理了评估生成器的逻辑。

1. 该部分首先进行生成器的前向传播。
2. 从模型输出中提取需要的信息，如音频特征 `x_ts`、先前的音频特征 `x_t_prevs`、先前的音频特征预测 `x_t_prev_preds`、说话人嵌入 `spk_emb` 和时间步 `t`。
3. 使用鉴别器对生成音频进行前向传播，得到条件真假音频的鉴别结果 `D_fake_cond`, `D_fake_uncond`。
4. 同样，对真实音频也进行鉴别器的前向传播，得到真实音频的鉴别结果 `D_real_cond`, `D_real_uncond`。
5. 调用生成器的损失函数 `Loss.g_loss_fn()`，计算生成音频的鉴别器损失 `adv_loss`。
6. 调用总的损失函数 `Loss()`，计算生成器的所有损失，包括声码器损失、重构损失、梅尔频谱损失、音高损失、能量损失和持续时间损失。
"""

                    G_loss = recon_loss + fm_loss + adv_loss

                losses = [D_loss + G_loss, D_loss, G_loss, recon_loss, fm_loss, adv_loss, mel_loss, pitch_loss, energy_loss, duration_loss]

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])
                        """在这段代码中：

- `G_loss` 是生成器的总损失，由重构损失 `recon_loss`、声码器损失 `fm_loss` 和对抗损失 `adv_loss` 组成。
- `losses` 是一个列表，包含了整个训练过程中的所有损失，包括了总的鉴别器损失、总的生成器损失以及各种子损失。
- 对 `losses` 中的每一个元素进行遍历，计算损失值的和，以便在训练过程中进行日志记录和可视化。
"""

    loss_means = []
    loss_means_msg = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_msg.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_msg.append(loss_sum / len(dataset))
    loss_means_msg = loss_means_msg[0:2] + loss_means_msg[5:]

    message = "Validation Step {}, Total Loss: {:.4f}, D_loss: {:.4f}, adv_loss: {:.4f}, mel_loss: {:.4f}, pitch_loss: {:.4f}, energy_loss: {:.4f}, duration_loss: {:.4f}".format(
        *([step] + [l for l in loss_means_msg])
    )
    """在这段代码中，对每个损失的总和进行了归一化，计算出了每个损失的平均值。然后，将这些平均值组成消息，用于在验证步骤中进行日志记录或显示。

以下是计算损失平均值并生成消息的代码解释：

1. `loss_means` 列表存储了每个损失的平均值。如果损失是字典类型，即包含了多个子损失，则计算每个子损失的平均值，并将结果存储在字典中；否则直接计算损失的平均值。
   
2. `loss_means_msg` 列表存储了用于日志记录或显示的损失消息。这里对每个损失的平均值进行了简单的处理，将字典类型的损失转换为总和，并去掉了不需要显示的损失项。

3. `message` 字符串包含了格式化后的损失消息，包括了验证步骤的编号以及各个损失项的平均值。
"""

    if logger is not None:
        figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            cond,
            args,
            batch,
            output,
            coarse_mels,
            vocoder,
            model_config,
            preprocess_config,
            model.module.diffusion,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            step,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/reconstructed",
        )
        log(
            logger,
            step,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/synthesized",
        )

    return message
"""在这段代码中：

- 首先，检查了是否传入了日志记录器 (`logger`)。如果没有，则不执行任何日志记录操作。
  
- 如果存在日志记录器，则调用 `synth_one_sample` 函数来合成一条样本，并生成音频和图像。

- 接着，使用 `log` 函数将损失值记录到日志中，并将图像和音频等附加信息一并记录，以便后续分析和评估。

- 最后，返回包含验证步骤信息和损失的消息字符串，以供需要时进行显示或进一步处理。
"""
