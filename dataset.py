import json
import math
import os

import torch
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D
from utils.pitch_tools import norm_interp_f0, get_lf0_cwt


class Dataset(Dataset):
    def __init__(
        self, filename, args, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.model = args.model
        self.preprocess_config = preprocess_config
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size" if self.model != "shallow" else "batch_size_shallow"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        """这段代码定义了一个自定义的数据集类 `Dataset`，该类继承自 PyTorch 的 `Dataset` 类。主要功能包括：

- **初始化方法 (`__init__`)：** 
  - 接受文件名、参数、预处理配置、模型配置和训练配置等参数。
  - 设置数据集相关的属性，如模型类型、预处理配置、数据集名称、预处理路径、文本清理器、批次大小等。
  - 处理元数据，包括提取基本名称、说话者、文本内容和原始文本等信息。
  - 加载说话者映射文件，以便后续使用。
  - 设置是否排序数据和是否丢弃最后一批数据的标志。

这个类的目的是提供一个灵活的数据集接口，可以根据参数配置加载和处理不同数据集的元数据，并提供用于训练和评估的数据。
"""

        # pitch stats
        self.pitch_type = preprocess_config["preprocessing"]["pitch"]["pitch_type"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            self.f0_mean = float(stats["f0"][0])
            self.f0_std = float(stats["f0"][1])

    def __len__(self):
        return len(self.text)
    """这部分代码在数据集类中添加了关于音高（pitch）统计信息的处理。具体来说：

- **初始化方法 (`__init__`)：** 
  - 从预处理配置中获取音高相关的信息，如音高类型。
  - 从预处理路径加载统计信息文件，包括音高均值和标准差。
  - 将统计信息存储在数据集对象中，以备后续使用。

- **`__len__` 方法：**
  - 返回数据集中文本的数量，即数据集的长度。

这部分代码的目的是为了提供音高统计信息的访问和使用，以便在数据预处理和模型训练过程中进行必要的标准化或其他处理。
"""

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        f0_path = os.path.join(
            self.preprocessed_path,
            "f0",
            "{}-f0-{}.npy".format(speaker, basename),
        )
        f0 = np.load(f0_path)
        f0, uv = norm_interp_f0(f0, self.preprocess_config["preprocessing"]["pitch"])
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        """这段代码是数据集类的 `__getitem__` 方法，用于根据索引 `idx` 获取数据集中的样本。具体来说：

- **提取基本信息：**
  - 获取指定索引处的基本信息，如文件名、说话者、原始文本等。
  - 根据索引提取相应的文本，并将其转换为序列（phones）。

- **加载特征：**
  - 构建用于加载声学特征的文件路径，包括梅尔频谱（mel）、音高（pitch）、基频（f0）和能量（energy）。
  - 加载对应的特征文件，并对基频进行归一化和插值处理。

这段代码的作用是根据索引加载数据集中的样本，并返回相应的信息以及声学特征。
"""
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        mel2ph_path = os.path.join(
            self.preprocessed_path,
            "mel2ph",
            "{}-mel2ph-{}.npy".format(speaker, basename),
        )
        mel2ph = np.load(mel2ph_path)
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None
        """这段代码继续加载了声学特征中的能量（energy）、语音帧时长（duration）和梅尔频谱到phones的对齐（mel2ph），并加载说话者嵌入（speaker embedding）。

- **加载能量和语音帧时长：**
  - 构建了能量和语音帧时长文件的路径，然后加载这些特征。

- **加载梅尔频谱到phones的对齐：**
  - 构建了梅尔频谱到phones的对齐文件的路径，然后加载该特征。

- **加载说话者嵌入：**
  - 如果配置中指定了使用说话者嵌入（speaker embedding），则加载对应的说话者嵌入文件，否则设置为None。
  """

        cwt_spec = f0_mean = f0_std = f0_ph = None
        if self.pitch_type == "cwt":
            cwt_spec_path = os.path.join(
                self.preprocessed_path,
                "cwt_spec",
                "{}-cwt_spec-{}.npy".format(speaker, basename),
            )
            cwt_spec = np.load(cwt_spec_path)
            f0cwt_mean_std_path = os.path.join(
                self.preprocessed_path,
                "f0cwt_mean_std",
                "{}-f0cwt_mean_std-{}.npy".format(speaker, basename),
            )
            f0cwt_mean_std = np.load(f0cwt_mean_std_path)
            f0_mean, f0_std = float(f0cwt_mean_std[0]), float(f0cwt_mean_std[1])
        elif self.pitch_type == "ph":
            f0_phlevel_sum = torch.zeros(phone.shape).float().scatter_add(
                0, torch.from_numpy(mel2ph).long() - 1, torch.from_numpy(f0).float())
            f0_phlevel_num = torch.zeros(phone.shape).float().scatter_add(
                0, torch.from_numpy(mel2ph).long() - 1, torch.ones(f0.shape)).clamp_min(1)
            f0_ph = (f0_phlevel_sum / f0_phlevel_num).numpy()
            """这部分代码处理了一些与语音的基频相关的特征。具体来说：

- 如果使用了连续小波变换（CWT）来表示基频特征：
  - 加载了CWT变换后的基频特征（cwt_spec）以及用于标准化的均值和标准差（f0_mean_std）。
- 如果使用了基频的phone级别对齐（ph）：
  - 计算了基频在每个phone级别上的平均值，并将其保存到f0_ph变量中。
  """

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "f0": f0,
            "f0_ph": f0_ph,
            "uv": uv,
            "cwt_spec": cwt_spec,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "energy": energy,
            "duration": duration,
            "mel2ph": mel2ph,
            "spker_embed": spker_embed,
        }

        return sample
    """这个函数返回一个样本字典，其中包含了每个样本的相关信息：

- "id": 样本的基本名称
- "speaker": 说话者的ID
- "text": 文本序列经过文本到序列的转换后的结果
- "raw_text": 未经处理的原始文本
- "mel": 梅尔频谱
- "pitch": 基频
- "f0": 经过标准化的基频
- "f0_ph": 在每个phone级别上的基频平均值
- "uv": 声门开合特征
- "cwt_spec": 连续小波变换后的基频特征
- "f0_mean": 基频的均值（用于标准化）
- "f0_std": 基频的标准差（用于标准化）
- "energy": 能量
- "duration": 持续时间
- "mel2ph": 梅尔频谱到phone级别的映射
- "spker_embed": 说话者嵌入特征

这些信息将在模型训练或推理过程中用于输入模型或者计算损失函数。
"""

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text
        
        """这个函数用于处理元数据文件，将文件中的信息提取出来并返回为四个列表：

- `name`: 基本名称列表
- `speaker`: 说话者ID列表
- `text`: 经过处理的文本列表
- `raw_text`: 原始文本列表

这些列表将用于构建数据集的基本信息。
"""

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        f0s = [data[idx]["f0"] for idx in idxs]
        uvs = [data[idx]["uv"] for idx in idxs]
        cwt_specs = f0_means = f0_stds = f0_phs = None
        if self.pitch_type == "cwt":
            cwt_specs = [data[idx]["cwt_spec"] for idx in idxs]
            f0_means = [data[idx]["f0_mean"] for idx in idxs]
            f0_stds = [data[idx]["f0_std"] for idx in idxs]
            cwt_specs = pad_2D(cwt_specs)
            f0_means = np.array(f0_means)
            f0_stds = np.array(f0_stds)
        elif self.pitch_type == "ph":
            f0s = [data[idx]["f0_ph"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        mel2phs = [data[idx]["mel2ph"] for idx in idxs]
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        """这个方法用于重新处理数据集中的样本，根据给定的索引列表。它从给定的数据中按索引提取出特定的样本信息，并进行必要的处理，然后返回处理后的样本信息。

具体来说，它从输入的`data`中根据`idxs`提取出指定索引对应的样本信息，包括基本信息如ID、说话者、文本等，以及与音频相关的信息如梅尔频谱、音高、能量、持续时间等。如果使用了cwt类型的音高处理，还会
提取cwt音高相关的信息。最后，它返回处理后的样本信息，以及文本长度和梅尔频谱长度的数组。
"""

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        f0s = pad_1D(f0s)
        uvs = pad_1D(uvs)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        mel2phs = pad_1D(mel2phs)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
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
        )
    """在重新处理后，为了确保样本具有相同的形状，可能需要对样本进行填充。这些填充操作通常在序列长度上进行，以确保样本对齐。在这个方法中，进行了以下填充操作：

- 对文本序列进行了一维填充。
- 对梅尔频谱序列进行了二维填充。
- 对音高、音高的标准差、能量、持续时间和梅尔频谱对应的位置进行了一维填充。

填充后，返回了填充后的所有样本信息，以及文本长度、梅尔频谱长度和各个维度的最大长度。
"""

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
    """这个 `collate_fn` 函数是用来处理批量加载的数据的。让我逐步解释它的作用：

1. 函数接受一个名为 `data` 的列表，其中每个元素代表一个样本。
2. 首先，它计算数据列表的大小 `data_size`。
3. 如果 `self.sort` 设置为 `True`，则对数据进行排序。它计算每个样本文本的长度，并根据长度从大到小进行排序。这样可以提高训练效率，因为在训练过程中通常希望每个批次的样本长度尽量接近，可以减少填充操作。
4. 接着，根据批次大小 `self.batch_size` 将数据索引分组。如果设置了 `drop_last=True` 并且剩余数据不足一个完整批次大小，则丢弃这部分数据。
5. 最后，对每个分组的数据进行重新处理（可能是填充操作等），然后将处理后的数据添加到输出列表 `output` 中。

总体来说，这个函数的主要作用是对数据进行排序（可选）、分组和重新处理，以便用于后续的训练过程。
"""


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        return (basename, speaker_id, phone, raw_text, spker_embed)
    """

1. `__init__` 方法：
   - `filepath`：指定包含文件名、说话人、文本和原始文本信息的文件路径。
   - `preprocess_config` 和 `model_config`：包含了预处理和模型配置信息的字典。

2. `__len__` 方法：
   - 返回数据集的长度，即数据集中样本的数量。

3. `__getitem__` 方法：
   - 根据给定的索引 `idx` 返回一个样本。
   - 从文件中加载了文件名、说话人、文本和原始文本信息。
   - 使用 `text_to_sequence` 函数将文本转换为对应的语音序列（例如音素序列）。
   - 如果需要，加载了与说话人相关的嵌入信息。

这个数据集类的作用是使得加载文本数据集变得更加方便，并为模型提供适当的输入。
"""

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds
    """这里是数据集类的另外两个重要方法：

1. `process_meta` 方法：
   - 从给定的文件中读取数据，并解析文件中的每一行，提取文件名、说话人、文本和原始文本信息。
   - 返回一个包含所有解析信息的元组 `(name, speaker, text, raw_text)`。

2. `collate_fn` 方法：
   - 在 DataLoader 中使用该方法来处理数据集中的批次数据。
   - 将数据按照批次组织，对于每个批次：
     - 提取批次中的所有样本的 ID、原始文本、说话人 ID、文本序列和文本长度。
     - 如果需要，加载了与说话人相关的嵌入信息。
     - 对文本序列进行填充，使得所有序列长度一致。
   - 返回一个包含批次数据的元组 `(ids, raw_texts, speakers, texts, text_lens, max_text_len, spker_embeds)`。
   """
