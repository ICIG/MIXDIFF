import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "LJSpeech"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
                    """这段代码是一个准备数据的函数，用于从给定的音频文件和相应的标注文本中提取并准备数据，以便后续的模型训练或其他处理。

具体步骤如下：

1. 从配置文件中读取输入目录路径（包含音频文件和元数据文件）、输出目录路径、采样率、最大音频值以及文本清洗器等参数。
2. 打开元数据文件（通常是一个 CSV 文件），逐行读取其中的内容。
3. 对于每一行，解析出文件名、文本内容等信息。
4. 清洗文本内容，根据配置文件中指定的文本清洗器进行处理。
5. 构造音频文件的完整路径，并检查文件是否存在。
6. 如果音频文件存在，首先创建输出目录（如果不存在），然后加载音频文件，并根据配置的最大音频值进行归一化处理，之后将音频文件写入到输出目录下的对应位置。
7. 将清洗后的文本内容写入到输出目录下相应文件的标注文件中。

这个函数的主要目的是将原始的音频文件和标注文本转换为模型可用的数据格式，以便后续的语音处理任务，例如文本转语音合成或语音识别。
"""