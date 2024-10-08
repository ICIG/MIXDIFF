import os
import re
import json
import argparse
from string import punctuation
from tqdm import tqdm

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples
from dataset import Dataset, TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""这行代码用于创建一个`device`对象，其类型为`torch.device`，用于指定模型在训练时所使用的设备，可以是CUDA GPU（如果可用）或CPU。如果CUDA可用，则选择GPU作为设备，否则选择CPU。
"""


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon
"""这个函数用于从指定的词典文件中读取词汇及其对应的音素。它接受一个词典文件路径作为输入参数，然后返回一个词典，其中包含了每个单词对应的音素列表。

函数首先创建一个空的词典 `lexicon`。然后，它打开指定路径的文件，并逐行读取文件内容。对于每一行，它使用正则表达式将单词和音素提取出来，然后将它们添加到 `lexicon` 中。如果一个单词在词典中已经存在，则只会更新它的音素列表。

最后，函数返回构建好的词典 `lexicon`。
"""


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


# def preprocess_mandarin(text, preprocess_config):
#     lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

#     phones = []
#     pinyins = [
#         p[0]
#         for p in pinyin(
#             text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
#         )
#     ]
#     for p in pinyins:
#         if p in lexicon:
#             phones += lexicon[p]
#         else:
#             phones.append("sp")

#     phones = "{" + " ".join(phones) + "}"
#     print("Raw Text Sequence: {}".format(text))
#     print("Phoneme Sequence: {}".format(phones))
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )

#     return np.array(sequence)
"""这个被注释的函数是用于对中文文本进行预处理，将其转换为对应的音素序列。它的处理流程与之前的英文预处理函数类似，但针对中文文本的特点做了一些调整。具体来说：

1. 读取预定义的词典文件，获取汉字和音素的映射关系。
2. 使用 `pinyin` 库将中文文本转换为拼音序列，其中使用了 `Style.TONE3` 参数来指定带有声调的拼音风格。
3. 对于每个拼音，检查它是否在词典中，如果在，则直接获取其对应的音素序列；如果不在，则将其标记为静音音素。
4. 将得到的音素序列用大括号括起来，并用空格分隔，以便后续处理。
5. 打印原始文本序列和音素序列。
6. 调用 `text_to_sequence` 函数将音素序列转换为数字序列，并返回该序列。

需要注意的是，该函数是被注释掉的，可能是因为当前环境不需要处理中文文本，或者有其他的原因导致暂时不需要使用。如果需要使用中文文本预处理功能，可以取消注释并调用该函数。
"""


def synthesize(model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    """这个 `synthesize` 函数的作用是进行语音合成，它接受以下参数：

- `model`：语音合成模型。
- `args`：命令行参数，可能包含一些配置信息。
- `configs`：包含预处理、模型和训练配置的元组。
- `vocoder`：声码器模型，用于将声学特征转换为语音波形。
- `batchs`：语音合成的批次数据。
- `control_values`：控制参数值，用于控制合成音频的音高、能量和持续时间。

函数首先从 `configs` 中解包出预处理、模型和训练配置。然后，根据提供的批次数据和控制参数值，利用模型进行语音合成。最终，使用声码器模型将合成的声学特征转换为最终的语音波形，并返回结果。
"""

    def synthesize_(batch):
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-1]),
                spker_embeds=batch[-1],
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )[0]
            synth_samples(
                args,
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                model.diffusion,
            )

    if args.teacher_forced:
        for batchs_ in batchs:
            for batch in tqdm(batchs_):
                batch = list(batch)
                batch[6] = None # set mel None for diffusion sampling
                synthesize_(batch)
    else:
        for batch in tqdm(batchs):
            synthesize_(batch)
            """这个 `synthesize_` 函数用于执行语音合成操作，它接收一个批次的数据作为输入。函数首先将批次数据移动到指定的设备（GPU 或 CPU），然后使用模型进行推理。在推理过程中，根据输入的批次数据和
        控制参数，生成合成的声学特征。接着，调用 `synth_samples` 函数，将生成的声学特征通过声码器模型转换为最终的语音波形，并保存合成的结果。最后，根据是否使用了 teacher forcing，循环遍历批次数据并
        进行合成操作。
            """


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=True,
        help="training model type",
    )
    """这段代码是一个命令行接口的配置部分。它使用 `argparse` 模块来解析命令行参数。具体来说：

- `restore_step` 参数用于指定从哪个训练步骤开始恢复训练。
- `path_tag` 参数是一个可选的字符串，用于向路径添加标记，以便在保存文件时进行区分。
- `model` 参数用于指定训练模型的类型，可以是 "naive"、"aux" 或 "shallow"。

在执行脚本时，用户需要通过命令行提供这些参数的值。
"""
    parser.add_argument("--teacher_forced", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    """这段代码是继续对命令行参数进行配置，具体如下：

- `teacher_forced` 参数是一个布尔值参数，当存在时，表示在合成过程中使用教师强制（teacher forcing）。
- `mode` 参数用于指定合成模式，可以是 "batch"（合成整个数据集）或 "single"（合成单个句子）。
- `source` 参数是一个字符串，用于指定批处理模式下数据集的路径，格式类似于 `train.txt` 和 `val.txt`。
- `text` 参数是一个字符串，用于指定单句合成模式下要合成的原始文本。

这些参数将允许用户在命令行中指定合成任务的不同配置。
"""
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="p225",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    """这部分代码添加了更多的命令行参数配置：

- `speaker_id` 参数用于多说话人合成时指定说话人的 ID。仅在单句合成模式下有效。
- `dataset` 参数用于指定数据集的名称。
- `pitch_control` 参数用于控制整个语音样本的音高，较大的值表示较高的音高。
"""
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()
    """这部分代码添加了更多的命令行参数配置：

- `energy_control` 参数用于控制整个语音样本的能量，较大的值表示较大的音量。
- `duration_control` 参数用于控制整个语音样本的语速，较大的值表示较慢的说话速率。

接下来，使用 `parser.parse_args()` 解析命令行参数并将结果存储在 `args` 变量中。
"""

    # Check source texts
    if args.mode == "batch":
        assert args.text is None
        if args.teacher_forced:
            assert args.source is None
        else:
            assert args.source is not None
    if args.mode == "single":
        assert args.source is None and args.text is not None and not args.teacher_forced
        """这段代码用于检查命令行参数的一致性。根据参数 `mode` 的取值，它会执行以下检查：

- 如果 `mode` 是 "batch"，则要求 `text` 参数为 `None`，如果使用了 `teacher_forced` 模式，则要求 `source` 参数也为 `None`，否则 `source` 参数不能为 `None`。
- 如果 `mode` 是 "single"，则要求 `source` 参数为 `None`，`text` 参数不为 `None`，并且不能使用 `teacher_forced` 模式。
"""

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    if args.model == "shallow":
        assert args.restore_step >= train_config["step"]["total_step_aux"]
    if args.model in ["aux", "shallow"]:
        train_tag = "shallow"
    elif args.model == "naive":
        train_tag = "naive"
    else:
        raise NotImplementedError
    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"]+"_{}{}".format(args.model, path_tag)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)
    """这段代码用于读取配置文件并根据命令行参数设置一些路径和标签。具体来说：

- `get_configs_of(args.dataset)` 用于根据给定的数据集名称获取预处理、模型和训练的配置。
- 然后根据命令行参数 `model` 的值设置训练标签 `train_tag`。如果 `model` 是 "shallow"，则要求 `restore_step` 大于等于辅助模型训练的总步数，否则会抛出错误。然后根据 `model` 的不同值更新一些路径，
包括检查点路径、日志路径和结果路径。
- 如果预处理配置中的音高类型是 "cwt"，则调用 `get_lf0_cwt` 函数获取 CWT 比例。
- 最后，根据 `restore_step` 创建结果路径。
"""

    # Log Configuration
    print("\n==================================== Inference Configuration ====================================")
    print(" ---> Type of Modeling:", args.model)
    print(" ---> Total Batch Size:", int(train_config["optimizer"]["batch_size"]))
    print(" ---> Path of ckpt:", train_config["path"]["ckpt_path"])
    print(" ---> Path of log:", train_config["path"]["log_path"])
    print(" ---> Path of result:", train_config["path"]["result_path"])
    print("================================================================================================")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    """这段代码用于打印推断的配置信息，获取模型并加载声码器。

- 首先，打印推断的配置信息，包括模型类型、总批大小以及检查点路径、日志路径和结果路径等信息。
- 然后，使用 `get_model` 函数获取模型，并将 `train` 参数设置为 `False`，以确保加载的是推断所需的模型。
- 最后，使用 `get_vocoder` 函数加载声码器，以便在推断过程中使用。
"""

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        if args.teacher_forced:
            dataset = Dataset(
                "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
            )
        else:
            dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
        """这部分代码用于预处理文本：

- 如果模式是批处理模式（`batch`），则根据情况创建数据集对象：
  - 如果使用了 `teacher_forced` 参数，则从验证集文件 `val.txt` 中获取数据集。
  - 否则，从指定的源文件 `args.source` 中获取数据集。
- 使用 `DataLoader` 加载数据集，设置批大小为 8，并指定 `collate_fn` 为数据集的 `collate_fn` 方法。
"""
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        
        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            raise NotImplementedError
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args, configs, vocoder, batchs, control_values)
    """这部分代码用于单句合成模式：

- 首先，根据给定的文本 `args.text`，生成一个包含文本 ID 和原始文本的列表。
- 如果模型配置为多说话人，并且使用了嵌入说话人信息，则从预处理路径中加载说话人映射和说话人嵌入。
- 根据文本语言执行相应的预处理操作：
  - 对于英文，使用 `preprocess_english` 函数进行预处理。
  - 对于中文，目前尚未实现对应的预处理功能，因此抛出 `NotImplementedError` 异常。
- 构建一个批次数据，包含文本 ID、原始文本、说话人 ID、预处理后的文本、文本长度等信息。
- 设置控制参数，包括音高控制、能量控制和时长控制。
- 调用 `synthesize` 函数进行合成。
"""
