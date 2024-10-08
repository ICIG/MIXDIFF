import os
import json
import yaml
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils.pitch_tools import denorm_f0, expand_f0_ph, cwt2f0


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config
"""这段代码定义了一个函数 `get_configs_of(dataset)`，用于获取指定数据集的配置信息。主要步骤如下：

1. 构建配置目录路径：根据提供的数据集名称，拼接配置目录的路径。

2. 加载预处理配置：从预处理配置文件中加载预处理参数。这里使用 `yaml` 模块的 `load` 函数加载配置文件，同时使用 `Loader=yaml.FullLoader` 指定加载器，以避免潜在的安全问题。

3. 加载模型配置：从模型配置文件中加载模型参数。

4. 加载训练配置：从训练配置文件中加载训练参数。

5. 返回加载得到的预处理配置、模型配置和训练配置。

需要注意的是，这段代码依赖于 `os` 和 `yaml` 模块，因此在使用之前需要确保这些模块已经导入。
"""


def to_device(data, device):
    if len(data) == 19:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            f0s,
            uvs,
            cwt_specs,
            f0_means,
            f0_stds,
            energies,
            durations,
            mel2phs,
            spker_embeds,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device) if mels is not None else mels
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).long().to(device)
        f0s = torch.from_numpy(f0s).float().to(device)
        uvs = torch.from_numpy(uvs).float().to(device)
        cwt_specs = torch.from_numpy(cwt_specs).float().to(device) if cwt_specs is not None else cwt_specs
        f0_means = torch.from_numpy(f0_means).float().to(device) if f0_means is not None else f0_means
        f0_stds = torch.from_numpy(f0_stds).float().to(device) if f0_stds is not None else f0_stds
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        mel2phs = torch.from_numpy(mel2phs).long().to(device)
        spker_embeds = torch.from_numpy(spker_embeds).float().to(device) if spker_embeds is not None else spker_embeds

        pitch_data = {
            "pitch": pitches,
            "f0": f0s,
            "uv": uvs,
            "cwt_spec": cwt_specs,
            "f0_mean": f0_means,
            "f0_std": f0_stds,
        }
        """这段代码定义了一个函数 `to_device(data, device)`，用于将数据移动到指定的设备上。该函数首先检查数据的长度是否为19，然后根据数据的不同部分，将其转换为 PyTorch 张量并将其移到指定的设备上。
        这个函数假设数据的结构是一个包含了19个元素的元组，每个元素对应着不同的数据。

接下来是对数据的转换和移动过程：

1. 将 `speakers`、`texts`、`src_lens`、`mel_lens`、`pitches`、`f0s`、`uvs`、`energies`、`durations` 和 `mel2phs` 转换为 PyTorch 的长整型张量，并移到指定设备上。

2. 将 `mels`、`cwt_specs` 和 `spker_embeds`（如果不为 None）转换为 PyTorch 的浮点型张量，并移到指定设备上。

3. 将 `f0_means` 和 `f0_stds`（如果不为 None）转换为 PyTorch 的浮点型张量，并移到指定设备上。

4. 最后，将转换后的 `pitches`、`f0s`、`uvs`、`cwt_specs`、`f0_means` 和 `f0_stds` 放入一个字典 `pitch_data` 中，并返回。

需要注意的是，该函数依赖于 `torch` 模块，因此在使用之前需要确保该模块已经导入。
"""

        return [
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitch_data,
            energies,
            durations,
            mel2phs,
            spker_embeds,
        ]

    if len(data) == 7:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds)
    """这段代码定义了一个条件语句，根据输入数据的长度来执行不同的操作。

1. 如果数据长度为19，则执行第一个分支，将数据按顺序封装到一个列表中，并返回。这个列表包含了数据的所有元素，以及之前定义的 `pitch_data` 字典。

2. 如果数据长度为7，则执行第二个分支，将数据按顺序封装到一个元组中，并返回。这个元组包含了数据的部分元素，不包括 `pitch_data`。

这样做的目的可能是为了根据不同的数据结构返回不同的数据形式。需要注意的是，这里使用了条件语句来判断数据的长度，所以输入数据的长度必须为19或者7。
"""


def log(
    logger, step=None, losses=None, lr=None, figs=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/D_loss", losses[1], step)
        logger.add_scalar("Loss/G_loss", losses[2], step)
        logger.add_scalar("Loss/recon_loss", losses[3], step)
        logger.add_scalar("Loss/fm_loss", losses[4], step)
        logger.add_scalar("Loss/adv_loss", losses[5], step)
        logger.add_scalar("Loss/mel_loss", losses[6], step)
        for k, v in losses[7].items():
            logger.add_scalar("Loss/{}_loss".format(k), v, step)
        logger.add_scalar("Loss/energy_loss", losses[8], step)
        for k, v in losses[9].items():
            logger.add_scalar("Loss/{}_loss".format(k), v, step)

    if lr is not None:
        logger.add_scalar("Training/learning_rate", lr, step)

    if figs is not None:
        for k, v in figs.items():
            logger.add_figure("{}/{}".format(tag, k), v, step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            step,
            sample_rate=sampling_rate,
        )
        """这段代码定义了一个名为 `log` 的函数，用于记录训练过程中的日志信息，如损失值、学习率、图像和音频等。

函数的参数包括：

- `logger`：日志记录器，可以是 TensorBoardX 或其他支持的日志记录工具。
- `step`：当前训练步数。
- `losses`：损失值，包括总损失、各种子损失。
- `lr`：学习率。
- `figs`：图像字典，用于记录图像。
- `audio`：音频信号，用于记录音频。
- `sampling_rate`：音频采样率。
- `tag`：标签，用于指定记录的日志名称。

函数首先检查每个输入参数是否为 `None`，然后使用 `logger` 对应的方法将信息记录下来。具体记录的内容包括：

1. 如果 `losses` 不为 `None`，则记录各种损失值。
2. 如果 `lr` 不为 `None`，则记录学习率。
3. 如果 `figs` 不为 `None`，则记录图像。
4. 如果 `audio` 不为 `None`，则记录音频。

这样设计的目的是方便记录训练过程中的各种信息，并通过日志可视化工具进行监控和分析。
"""


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)
"""这段代码包含两个函数：

1. `get_mask_from_lengths(lengths, max_len=None)` 函数根据输入的长度数组生成一个掩码张量，用于在序列数据中标记有效的位置。主要步骤如下：
   - 首先获取批次大小（batch_size），即长度数组的形状的第一个维度。
   - 如果未提供 `max_len` 参数，则将 `max_len` 设为长度数组中最大的值。
   - 创建一个索引张量 `ids`，其形状为 `(batch_size, max_len)`，每一行都是从 0 到 `max_len-1` 的整数序列。
   - 生成一个掩码张量 `mask`，其形状与 `ids` 相同，其中的元素值为布尔型，表示对应位置是否为无效位置（即超出了有效长度）。
   
2. `expand(values, durations)` 函数根据持续时间数组来扩展值数组，使得每个值重复相应的持续时间次数。具体步骤如下：
   - 遍历输入的值数组和持续时间数组，对于每个值和持续时间，重复该值相应的持续时间次数，并将重复后的值添加到输出列表中。
   - 最后将输出列表转换为 NumPy 数组并返回。

这两个函数都是用于序列数据处理的辅助函数，可能用于文本处理、语音处理或其他序列数据的任务中。需要注意的是，这里的 `device` 变量在函数中未定义，需要确保其在函数所在的作用域中已经定义。
"""


def synth_one_sample(cond,args, targets, predictions, coarse_mels, vocoder, model_config, preprocess_config, diffusion):

    pitch_config = preprocess_config["preprocessing"]["pitch"]
    pitch_type = pitch_config["pitch_type"]
    use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
    use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
    timesteps = model_config["denoiser"]["timesteps" if args.model == "naive" else "shallow_timesteps"]
    basename = targets[0][0]
    src_len = predictions[10][0].item()
    mel_len = predictions[11][0].item()
    mel_target = targets[6][0, :mel_len].float().detach().transpose(0, 1)
    duration = targets[11][0, :src_len].int().detach().cpu().numpy()
    figs = {}
    if use_pitch_embed:
        pitch_prediction, pitch_target = predictions[4], targets[9]
        f0 = pitch_target["f0"]
        if pitch_type == "ph":
            mel2ph = targets[12]
            f0 = expand_f0_ph(f0, mel2ph, pitch_config)
            f0_pred = expand_f0_ph(pitch_prediction["pitch_pred"][:, :, 0], mel2ph, pitch_config)
            figs["f0"] = f0_to_figure(f0[0, :mel_len], None, f0_pred[0, :mel_len])
        else:
            f0 = denorm_f0(f0, pitch_target["uv"], pitch_config)
            if pitch_type == "cwt":
                # cwt
                cwt_out = pitch_prediction["cwt"]
                cwt_spec = cwt_out[:, :, :10]
                cwt = torch.cat([cwt_spec, pitch_target["cwt_spec"]], -1)
                figs["cwt"] = spec_to_figure(cwt[0, :mel_len])
                # f0
                f0_pred = cwt2f0(cwt_spec, pitch_prediction["f0_mean"], pitch_prediction["f0_std"], pitch_config["cwt_scales"])
                if pitch_config["use_uv"]:
                    assert cwt_out.shape[-1] == 11
                    uv_pred = cwt_out[:, :, -1] > 0
                    f0_pred[uv_pred > 0] = 0
                f0_cwt = denorm_f0(pitch_target["f0_cwt"], pitch_target["uv"], pitch_config)
                figs["f0"] = f0_to_figure(f0[0, :mel_len], f0_cwt[0, :mel_len], f0_pred[0, :mel_len])
            elif pitch_type == "frame":
                # f0
                uv_pred = pitch_prediction["pitch_pred"][:, :, 1] > 0
                pitch_pred = denorm_f0(pitch_prediction["pitch_pred"][:, :, 0], uv_pred, pitch_config)
                figs["f0"] = f0_to_figure(f0[0, :mel_len], None, pitch_pred[0, :mel_len])
                """这段代码是用于合成一个样本的声音波形，并生成一些用于可视化的图像。

主要步骤如下：

1. 从输入参数中获取必要的配置信息和预测结果。

2. 根据预测结果和目标值，提取所需的信息，如目标的声学特征（mel频谱）、持续时间等。

3. 如果模型中使用了音高（pitch）嵌入（use_pitch_embed=True），则根据配置和预测结果，生成音高相关的图像。这包括了不同类型的音高（基频）预测和目标的对比图。

4. 如果模型中使用了能量（energy）嵌入（use_energy_embed=True），可能会执行类似的操作，但在代码中并未给出。

5. 如果需要，根据预测结果，生成CWT（连续小波变换）和相应的音高（基频）图像。

6. 最后，返回所生成的图像字典。

需要注意的是，这里使用了一些辅助函数，如 `expand_f0_ph`、`f0_to_figure`、`spec_to_figure`、`cwt2f0`、`denorm_f0` 等，这些函数可能是用于处理音频相关的数据、图像转换等操作。
"""
    if use_energy_embed:
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy_prediction = predictions[5][0, :src_len].detach().cpu().numpy()
            energy_prediction = expand(energy_prediction, duration)
            energy_target = targets[10][0, :src_len].detach().cpu().numpy()
            energy_target = expand(energy_target, duration)
        else:
            energy_prediction = predictions[5][0, :mel_len].detach().cpu().numpy()
            energy_target = targets[10][0, :mel_len].detach().cpu().numpy()
        figs["energy"] = energy_to_figure(energy_target, energy_prediction)

    if args.model == "aux":
        # denormalizing x_0 is needed due to diffuse_trace
        mel_prediction = diffusion.denorm_spec(predictions[0])[0, :mel_len].float().detach().transpose(0, 1)
        mels = [
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
        ]
        titles = ["Sampled Spectrogram", "GT"]
    else:
        mels = [mel_pred[0, :mel_len].float().detach().transpose(0, 1) for mel_pred in diffusion.sampling(cond=cond)]
        mel_prediction = mels[-1]
        if args.model == "shallow":
            coarse_mel = coarse_mels[0, :mel_len].float().detach().transpose(0, 1)
            mels.append(coarse_mel)
        mels.append(mel_target)
        titles = [f"T={t}" if t!=0 else f"T={t}" for t in range(0, timesteps+1)[::-1]] \
            + (["Coarse Spectrogram"] if args.model == "shallow" else []) + ["GT"]
        diffusion.aux_mel = None

    figs["mel"] = plot_mel(mels, titles)
    """在原始代码中，添加了一些新的逻辑：

1. 如果模型中使用了能量（energy）嵌入（`use_energy_embed=True`），则根据配置和预测结果生成能量相关的图像。具体操作包括：
   - 检查预处理配置中的能量特征级别（feature），如果是 "phoneme_level"，则将能量预测和目标扩展为与持续时间匹配的长度，然后生成能量图像。
   - 否则，将能量预测和目标扩展为与 mel 长度匹配的长度，然后生成能量图像。

2. 如果模型类型是 "aux"，则：
   - 对于扩散模型，需要将预测的 mel 频谱转换为正常的频谱并生成图像。
   - 将生成的 mel 频谱与目标 mel 频谱以及（如果存在）粗糙 mel 频谱添加到一个列表中。
   - 根据不同的时间步长（T），生成不同的标题。
   
3. 如果模型类型不是 "aux"，则：
   - 对于 "shallow" 模型，需要将粗糙 mel 频谱添加到列表中。
   - 对于其他类型的模型，需要根据条件生成多个 mel 频谱，并将它们添加到列表中。

4. 最后，生成 mel 频谱的图像，并将其添加到 `figs` 字典中。

这些逻辑用于根据模型类型和配置生成相应的图像，以便在训练过程中对模型的性能进行评估和调试。
"""

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return figs, wav_reconstruction, wav_prediction, basename
"""在这段代码中，根据是否提供了声码器（vocoder）对象，进行不同的处理：

1. 如果提供了声码器对象（`vocoder is not None`），则：
   - 导入声码器推断函数。
   - 使用声码器对目标 mel 频谱和预测 mel 频谱进行声音重构，得到重构的音频和预测的音频。
   - 将重构的音频和预测的音频分别赋值给 `wav_reconstruction` 和 `wav_prediction`。

2. 如果没有提供声码器对象，则 `wav_reconstruction` 和 `wav_prediction` 均为 None。

最后，返回生成的图像字典 `figs`、重构的音频 `wav_reconstruction`、预测的音频 `wav_prediction` 以及基准名称 `basename`。这些结果可能用于评估模型的性能和生成可视化结果。
"""


def synth_samples(args, targets, predictions, vocoder, model_config, preprocess_config, path, diffusion):

    multi_speaker = model_config["multi_speaker"]
    teacher_forced_tag = "_teacher_forced" if args.teacher_forced else ""
    basenames = targets[0]
    if args.model == "aux":
        # denormalizing x_0 is needed due to diffuse_trace
        predictions[0] = diffusion.denorm_spec(predictions[0][0])
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[10][i].item()
        mel_len = predictions[11][i].item()
        mel_prediction = predictions[0][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[7][i, :src_len].detach().cpu().numpy()

        fig_save_dir = os.path.join(
            path, str(args.restore_step), "{}_{}{}.png".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.png".format(basename, teacher_forced_tag))
        fig = plot_mel(
            [
                mel_prediction.cpu().numpy(),
            ],
            ["Synthetized Spectrogram"],
        )
        plt.savefig(fig_save_dir)
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[0].transpose(1, 2)
    lengths = predictions[11] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(
            path, str(args.restore_step), "{}_{}{}.wav".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.wav".format(basename, teacher_forced_tag)),
            sampling_rate, wav)
        """这段代码定义了一个函数 `synth_samples`，用于合成多个样本的音频和频谱图像。

函数的主要步骤如下：

1. 获取模型配置中的一些参数，例如是否是多说话人模型（multi_speaker）、是否是教师强制（teacher_forced）等。

2. 遍历每个样本的预测结果，对于每个样本，生成对应的频谱图像，并将其保存到指定路径下。图像的标题为 "Synthetized Spectrogram"。

3. 使用声码器对所有样本的 mel 频谱进行声音合成，得到对应的音频数据。

4. 将合成的音频数据保存为 WAV 文件，文件名包括样本的基准名称和一些额外信息（如果适用）。

需要注意的是，这里的声码器推断函数 `vocoder_infer` 被用来对 mel 频谱进行声音合成。同时，这里也用到了一些配置信息，如采样率（sampling_rate）等。
"""


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, figsize=(8, len(data) * 4), squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig
"""这段代码定义了一个函数 `plot_mel`，用于绘制 mel 频谱图像。

函数接受两个参数：
- `data`：一个包含 mel 频谱数据的列表或张量。如果传入的是张量，会将其转换为 NumPy 数组。
- `titles`：一个可选参数，包含每个 mel 频谱图像的标题。如果未提供，则默认为 None。

函数首先创建一个包含多个子图的图像对象，然后遍历输入的 mel 数据，对每个 mel 频谱进行如下操作：
- 将 mel 数据转换为 NumPy 数组（如果是张量）。
- 使用 `imshow` 方法将 mel 数据绘制为图像。
- 设置图像的纵轴范围为 0 到 mel 数据的行数。
- 设置图像的标题（如果提供了标题）。
- 调整图像的外观和布局，使其更易读。

最后返回绘制好的图像对象。
"""


def plot_embedding(out_dir, embedding, embedding_speaker_id, gender_dict, filename='embedding.png'):
    colors = 'r','b'
    labels = 'Female','Male'

    data_x = embedding
    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()
    """这段代码定义了一个函数 `plot_embedding`，用于绘制嵌入空间的可视化图。

函数接受以下参数：

- `out_dir`：输出目录，用于保存生成的图像文件。
- `embedding`：嵌入向量数据。
- `embedding_speaker_id`：嵌入向量对应的说话者 ID。
- `gender_dict`：一个字典，将说话者 ID 映射到其性别。
- `filename`：可选参数，保存的图像文件名，默认为 'embedding.png'。

函数的主要步骤如下：

1. 定义了两种颜色（红色和蓝色）和对应的标签（女性和男性）。
2. 使用 t-SNE 算法将嵌入向量降维到二维空间。
3. 根据性别信息，将嵌入向量在二维空间中绘制为散点图，颜色代表性别。
4. 绘制图像并保存到指定的输出目录中。

最后，生成的图像文件保存在指定的输出目录中，并关闭当前的图像绘制环境。
"""


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure()
    if isinstance(f0_gt, torch.Tensor):
        f0_gt = f0_gt.detach().cpu().numpy()
    plt.plot(f0_gt, color="r", label="gt")
    if f0_cwt is not None:
        if isinstance(f0_cwt, torch.Tensor):
            f0_cwt = f0_cwt.detach().cpu().numpy()
        plt.plot(f0_cwt, color="b", label="cwt")
    if f0_pred is not None:
        if isinstance(f0_pred, torch.Tensor):
            f0_pred = f0_pred.detach().cpu().numpy()
        plt.plot(f0_pred, color="green", label="pred")
    plt.legend()
    return fig
"""这段代码定义了两个函数，分别用于绘制频谱图和音高图的可视化图像。

1. `spec_to_figure(spec, vmin=None, vmax=None)` 函数用于绘制频谱图像，接受以下参数：
   - `spec`：频谱数据，可以是 NumPy 数组或 PyTorch 张量。
   - `vmin`：可选参数，指定颜色映射的最小值。
   - `vmax`：可选参数，指定颜色映射的最大值。
   函数首先将输入的频谱数据转换为 NumPy 数组，然后创建一个图像对象，并使用 `pcolor` 方法绘制频谱图，最后返回图像对象。

2. `f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None)` 函数用于绘制音高图像，接受以下参数：
   - `f0_gt`：真实音高数据，可以是 NumPy 数组或 PyTorch 张量。
   - `f0_cwt`：CWT 方法预测的音高数据，可选参数，可以是 NumPy 数组或 PyTorch 张量。
   - `f0_pred`：模型预测的音高数据，可选参数，可以是 NumPy 数组或 PyTorch 张量。
   函数首先将输入的音高数据转换为 NumPy 数组，然后创建一个图像对象，并使用 `plot` 方法绘制真实音高数据。如果提供了 CWT 预测或模型预测的音高数据，也将其绘制在图像上，并添加相应的图例，最后返回
   图像对象。

这两个函数用于绘制频谱和音高的可视化图像，便于分析和可视化模型的输出结果。
"""


def energy_to_figure(energy_gt, energy_pred=None):
    fig = plt.figure()
    if isinstance(energy_gt, torch.Tensor):
        energy_gt = energy_gt.detach().cpu().numpy()
    plt.plot(energy_gt, color="r", label="gt")
    if energy_pred is not None:
        if isinstance(energy_pred, torch.Tensor):
            energy_pred = energy_pred.detach().cpu().numpy()
        plt.plot(energy_pred, color="green", label="pred")
    plt.legend()
    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded
"""这段代码定义了两个函数：

1. `energy_to_figure(energy_gt, energy_pred=None)` 函数用于绘制能量图像，接受以下参数：
   - `energy_gt`：真实能量数据，可以是 NumPy 数组或 PyTorch 张量。
   - `energy_pred`：模型预测的能量数据，可选参数，可以是 NumPy 数组或 PyTorch 张量。
   函数首先将输入的能量数据转换为 NumPy 数组，然后创建一个图像对象，并使用 `plot` 方法绘制真实能量数据。如果提供了模型预测的能量数据，也将其绘制在图像上，并添加相应的图例，最后返回图像对象。

2. `pad_1D(inputs, PAD=0)` 函数用于对一维输入数据进行填充，接受以下参数：
   - `inputs`：一个包含一维数据的列表或数组。
   - `PAD`：填充值，默认为 0。
   函数首先计算输入数据中最长序列的长度，然后对所有序列进行填充，使它们的长度保持一致，最后返回填充后的数组。

这两个函数可以用于处理能量数据和一维序列数据，例如进行填充或绘制图像。
"""


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
"""这段代码定义了两个函数，用于对二维输入数据进行填充：

1. `pad_2D(inputs, maxlen=None)` 函数用于对二维输入数据进行填充，接受以下参数：
   - `inputs`：一个包含二维数据的列表或数组。
   - `maxlen`：可选参数，指定填充后的最大长度。
   函数首先遍历所有输入数据，找到最长的列数（即二维数组的第二维度），然后将所有数组的行数进行填充，使它们的行数保持一致，最后返回填充后的二维数组。

2. `pad(input_ele, mel_max_length=None)` 函数用于对 PyTorch 张量的输入进行填充，接受以下参数：
   - `input_ele`：一个包含 PyTorch 张量的列表。
   - `mel_max_length`：可选参数，指定填充后的最大长度。
   函数首先计算输入张量中最长序列的长度，然后对所有张量进行填充，使它们的长度保持一致，最后返回填充后的张量。

这两个函数用于在进行批量处理时，确保所有输入数据的维度一致，便于模型的处理。
"""


def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)


def get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_mode == "vpsde":
        schedule_list = np.array([
            vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
    else:
        raise NotImplementedError
    return schedule_list
"""这段代码定义了两个函数：

1. `vpsde_beta_t(t, T, min_beta, max_beta)` 函数用于计算可变概率采样分布中的 beta 参数值，接受以下参数：
   - `t`：当前时间步。
   - `T`：总时间步数。
   - `min_beta`：beta 参数的最小值。
   - `max_beta`：beta 参数的最大值。
   函数首先计算一个与时间步和总时间步数相关的系数 `t_coef`，然后根据给定的最小值、最大值和系数计算并返回 beta 参数值。

2. `get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0, max_beta=0.01, s=0.008)` 函数用于生成噪声调度列表，接受以下参数：
   - `schedule_mode`：调度模式，可以是 "linear"、"cosine" 或 "vpsde"。
   - `timesteps`：总时间步数。
   - `min_beta`：beta 参数的最小值，默认为 0.0。
   - `max_beta`：beta 参数的最大值，默认为 0.01。
   - `s`：余弦调度模式中的参数，默认为 0.008。
   根据指定的调度模式，生成相应的噪声调度列表，并返回。

这两个函数用于生成在可变概率采样过程中使用的参数值或噪声调度列表，用于控制采样过程的温度和噪声。
"""


def dur_to_mel2ph(dur, dur_padding=None, alpha=1.0):
    """
    Example (no batch dim version):
        1. dur = [2,2,3]
        2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
        3. token_mask = [[1,1,0,0,0,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,1,1]]
        4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                        [0,0,2,2,0,0,0],
                                        [0,0,0,0,3,3,3]]
        5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

    :param dur: Batch of durations of each frame (B, T_txt)
    :param dur_padding: Batch of padding of each frame (B, T_txt)
    :param alpha: duration rescale coefficient
    :return:
        mel2ph (B, T_speech)
    """
    assert alpha > 0
    dur = torch.round(dur.float() * alpha).long()
    if dur_padding is not None:
        dur = dur * (1 - dur_padding.long())
    token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
    dur_cumsum = torch.cumsum(dur, 1)
    dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

    pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
    token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
    mel2ph = (token_idx * token_mask.long()).sum(1)
    return mel2ph
"""这段代码定义了一个函数 `dur_to_mel2ph`，用于将持续时间转换为音素位置索引。这个函数的输入是持续时间序列，它会返回相应的音素位置索引序列。

函数的主要步骤如下：

1. 将输入的持续时间乘以一个尺度系数 `alpha` 并取整，以确保结果为整数类型。
2. 如果提供了持续时间的填充信息 `dur_padding`，则将持续时间乘以相应位置的填充掩码（1 减去填充值）。
3. 创建一个索引张量 `token_idx`，其中每个元素表示一个音素位置的索引。
4. 计算累积持续时间序列 `dur_cumsum` 和前一个时间步的累积持续时间序列 `dur_cumsum_prev`。
5. 创建一个位置索引张量 `pos_idx`，其中包含从 0 开始到最大持续时间之间的所有位置索引。
6. 基于持续时间的累积值，为每个位置确定其所属的音素索引，并将结果保存在 `mel2ph` 中。

最终，函数返回一个音素位置索引序列 `mel2ph`，其形状为 (B, T_speech)，其中 B 是批量大小，T_speech 是语音的总帧数。
"""


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx
"""这段代码定义了两个函数：

1. `mel2ph_to_dur(mel2ph, T_txt, max_dur=None)` 函数用于将音素位置索引转换为持续时间。它接受以下参数：
   - `mel2ph`：音素位置索引张量，形状为 (B, T_speech)，其中 B 是批量大小，T_speech 是语音的总帧数。
   - `T_txt`：文本序列的长度。
   - `max_dur`：可选参数，最大持续时间。如果提供了此参数，则会对持续时间进行截断，使其不超过最大值。
   函数首先创建一个与 `mel2ph` 相同大小的零张量，并根据音素位置索引对其进行累加。然后，将第一列删除以得到最终的持续时间张量。

2. `make_positions(tensor, padding_idx)` 函数用于替换非填充符号的位置编号。它接受以下参数：
   - `tensor`：输入张量，其中填充符号用于表示不相关的位置。
   - `padding_idx`：填充符号的索引。
   函数首先创建一个掩码张量，其中非填充符号的位置用 1 表示，填充符号用 0 表示。然后，使用 `cumsum` 函数计算每个位置的位置编号，最后将填充符号的位置编号设置为填充符号的索引。

这两个函数主要用于在文本到语音的转换过程中处理持续时间信息，并且可以用于模型训练和推理过程中的不同阶段。
"""


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)
    """这段代码实现了 SSIM（Structural Similarity Index）算法，用于评估两个图像的结构相似性。以下是代码中每个函数的作用：

1. `gaussian(window_size, sigma)`：生成一个一维高斯窗口，用于平滑图像。它接受两个参数：
   - `window_size`：窗口大小。
   - `sigma`：高斯函数的标准差。
   函数首先计算窗口中每个位置的高斯值，然后对其进行归一化，以确保总和为 1，并返回结果。

2. `create_window(window_size, channel)`：创建一个二维的高斯窗口。它接受两个参数：
   - `window_size`：窗口大小。
   - `channel`：图像的通道数。
   函数首先调用 `gaussian` 函数创建一个一维高斯窗口，然后对其进行外积操作，得到二维高斯窗口，并在通道维度上进行扩展，最后返回结果。

3. `_ssim(img1, img2, window, window_size, channel, size_average=True)`：计算两个图像之间的 SSIM 值。它接受以下参数：
   - `img1`：第一个图像。
   - `img2`：第二个图像。
   - `window`：用于加权的窗口。
   - `window_size`：窗口大小。
   - `channel`：图像的通道数。
   - `size_average`：是否对结果进行平均。如果为 `True`，则返回全局的 SSIM 值；如果为 `False`，则返回每个像素点的 SSIM 值。
   函数首先通过卷积操作计算图像的均值和方差，然后利用这些值计算 SSIM 映射，最后根据 `size_average` 参数决定返回结果的类型。
   """


def ssim(img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        global window
        if window is None:
            window = create_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)
"""这是一个用于计算 SSIM（Structural Similarity Index）的函数。它接受两个图像作为输入，并返回它们之间的 SSIM 值。以下是函数的参数和工作原理：

- `img1`：第一个图像。
- `img2`：第二个图像。
- `window_size`：用于计算 SSIM 的窗口大小，默认为 11。
- `size_average`：是否对结果进行平均。如果为 `True`，则返回全局的 SSIM 值；如果为 `False`，则返回每个像素点的 SSIM 值。

函数首先检查 `window` 是否已经创建，如果没有，则调用 `create_window` 函数创建一个二维的高斯窗口，并根据图像的通道数进行扩展。然后，调用 `_ssim` 函数计算图像之间的 SSIM 值，并根据
 `size_average` 参数决定返回结果的类型。
"""
