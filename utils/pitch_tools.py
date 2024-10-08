#########
# world
#########
import librosa
import parselmouth
import numpy as np
import torch
import torch.nn.functional as F
from pycwt import wavelet
from scipy.interpolate import interp1d

gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)
FFT_SIZE = 2048
"""这些变量是用于音频信号处理的参数。让我为您解释它们：

- `gamma`: 控制梅尔倒谱系数（MFCC）的功率谱的平滑度。通常情况下，较低的值会导致更多的高频谱信息被保留。
- `mcepInput`: 控制输入到声码器的声学特征的类型。在这种情况下，`3` 表示输入的是幅度谱（magnitude spectrum）。
- `alpha`: 预加重滤波器的系数，用于提高高频信号的能量以改善语音质量。
- `en_floor`: 能量的最小阈值，通常用于稳定声学模型的训练过程。
- `FFT_SIZE`: 用于傅里叶变换的窗口大小，决定了频谱分析的精度和频率分辨率。在这种情况下，窗口大小为 2048。
"""


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)
"""这些变量是用于基频（F0）处理的参数。让我解释它们：

- `f0_bin`: F0 的分辨率，即在提取 F0 时使用的离散级数的数量。在这里，设置为 256，表示将 F0 范围等分成 256 个部分。
- `f0_max` 和 `f0_min`: F0 的最大和最小允许值。在这里，`f0_max` 设置为 1100 Hz，`f0_min` 设置为 50 Hz。
- `f0_mel_min` 和 `f0_mel_max`: 将最大和最小 F0 转换为梅尔刻度的值。梅尔刻度是一种对音调感知的非线性度量，通常用于表示声音高度。
"""


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def norm_f0(f0, uv, config):
    is_torch = isinstance(f0, torch.Tensor)
    if config["pitch_norm"] == "standard":
        f0 = (f0 - config["f0_mean"]) / config["f0_std"]
    if config["pitch_norm"] == "log":
        eps = config["pitch_norm_eps"]
        f0 = torch.log2(f0 + eps) if is_torch else np.log2(f0 + eps)
    if uv is not None and config["use_uv"]:
        f0[uv > 0] = 0
    return f0
"""这些函数用于对 F0 进行预处理和量化：

- `f0_to_coarse`: 将原始 F0 转换为粗糙的离散级数。首先将 F0 转换为梅尔刻度空间，然后根据预定义的 `f0_bin` 参数对其进行量化。这样可以将连续的 F0 值映射到离散的级数上，以便于模型处理。
- `norm_f0`: 对 F0 进行归一化处理。根据配置中的规范化方式，可以选择使用标准化或对数化。标准化会将 F0 调整为零均值和单位方差，而对数化会将 F0 取对数。此外，如果使用了声门开闭信息（uv），则可以根据需要将声音置零。

这些函数能够将原始的 F0 数据转换为模型所需的粗糙表示，并根据需要对其进行归一化处理。
"""


def norm_interp_f0(f0, config):
    # is_torch = isinstance(f0, torch.Tensor)
    # if is_torch:
    #     device = f0.device
    #     f0 = f0.data.cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, config)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    # uv = torch.FloatTensor(uv)
    # f0 = torch.FloatTensor(f0)
    # if is_torch:
    #     f0 = f0.to(device)
    return f0, uv


def denorm_f0(f0, uv, config, pitch_padding=None, min=None, max=None):
    if config["pitch_norm"] == "standard":
        f0 = f0 * config["f0_std"] + config["f0_mean"]
    if config["pitch_norm"] == "log":
        f0 = 2 ** f0
    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and config["use_uv"]:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0
"""这两个函数用于在 F0 数据上执行归一化和反归一化操作：

- `norm_interp_f0`: 对 F0 数据执行归一化和插值操作。首先，将 F0 数据进行归一化处理，然后根据未标记的声门区域（非声音）对 F0 进行插值。这一步操作用于填补未标记的声音区域。最后，将处理后的 F0 数据
与声门标记一起返回。
  
- `denorm_f0`: 对归一化后的 F0 数据执行反归一化操作。根据配置中的规范化方式，执行标准化或反对数化。此外，还可以对 F0 进行截断或填充操作，以满足特定的范围或条件。
"""


def get_pitch(wav_data, mel, config):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param config:
    :return:
    """
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    time_step = hop_length / sampling_rate * 1000
    f0_min = 80
    f0_max = 750

    if hop_length == 128:
        pad_size = 4
    elif hop_length  == 256:
        pad_size = 2
    else:
        assert False

    f0 = parselmouth.Sound(wav_data, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array["frequency"]
    f0 = f0[:len(mel)-8] # to avoid negative rpad
    lpad = pad_size * 2
    rpad = len(mel) - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")
    # mel and f0 are extracted by 2 different libraries. we should force them to have the same length.
    # Attention: we find that new version of some libraries could cause ``rpad'' to be a negetive value...
    # Just to be sure, we recommend users to set up the same environments as them in requirements_auto.txt (by Anaconda)
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse


def expand_f0_ph(f0, mel2ph, config):
    f0 = denorm_f0(f0, None, config)
    f0 = F.pad(f0, [1, 0])
    f0 = torch.gather(f0, 1, mel2ph)  # [B, T_mel]
    return f0
"""这两个函数用于处理声调（pitch）数据：

- `get_pitch`: 该函数根据音频数据提取声调数据。首先，使用 Parselmouth 库将音频数据转换为声调数据，并设置最小和最大频率。然后，对于一致的长度，使用填充来调整声调数据和 Mel 频谱的长度。最后，将声调
数据转换为粗糙的离散值。

- `expand_f0_ph`: 此函数用于将声调数据扩展到相应的 Mel 频谱上。首先，反归一化声调数据。然后，通过填充操作将其调整为与 Mel 频谱相同的长度。最后，根据 Mel 与声音（pitch）对齐信息，将声调数据扩展到
相应的 Mel 频谱上。
"""


#########
# cwt
#########


def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0
    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    f0 = np.copy(f0)
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        print("| all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0
"""这两个函数用于处理音频数据：

- `load_wav`: 该函数用于加载音频文件并返回音频数据。它使用 librosa 库加载音频文件，并根据指定的采样率对音频数据进行重采样。

- `convert_continuos_f0`: 此函数用于将离散的声调（F0）序列转换为连续的声调序列。首先，根据离散声调序列，获取非零声调帧的索引。然后，通过线性插值获得连续的声调序列。
"""


def get_cont_lf0(f0, frame_period=5.0):
    uv, cont_f0_lpf = convert_continuos_f0(f0)
    # cont_f0_lpf = low_pass_filter(cont_f0_lpf, int(1.0 / (frame_period * 0.001)), cutoff=20)
    cont_lf0_lpf = np.log(cont_f0_lpf)
    return uv, cont_lf0_lpf


def get_lf0_cwt(lf0):
    """
    input:
        signal of shape (N)
    output:
        Wavelet_lf0 of shape(10, N), scales of shape(10)
    """
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 2
    J = 9

    Wavelet_lf0, scales, _, _, _, _ = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    # Wavelet.shape => (J + 1, len(lf0))
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0, scales


def norm_scale(Wavelet_lf0):
    Wavelet_lf0_norm = np.zeros((Wavelet_lf0.shape[0], Wavelet_lf0.shape[1]))
    mean = Wavelet_lf0.mean(0)[None, :]
    std = Wavelet_lf0.std(0)[None, :]
    Wavelet_lf0_norm = (Wavelet_lf0 - mean) / std
    return Wavelet_lf0_norm, mean, std
"""这些函数用于对声调（F0）进行进一步的处理：

- `get_cont_lf0`: 此函数将离散的声调序列转换为对数频率（LF0）序列。它首先使用 `convert_continuos_f0` 函数将离散的声调序列转换为连续的声调序列，然后取对数以获得连续的 LF0 序列。

- `get_lf0_cwt`: 该函数用于计算 LF0 序列的连续小波变换（CWT）。它使用小波分析来捕获信号的时间-频率特征。具体地，它使用墨西哥帽小波作为小波函数，计算给定 LF0 序列的连续小波变换，并返回变换结果以及
相应的尺度。

- `norm_scale`: 此函数用于归一化 LF0 序列的小波变换结果。它对每个尺度的 LF0 序列应用零均值单位方差（Z-score）归一化。
"""


def normalize_cwt_lf0(f0, mean, std):
    uv, cont_lf0_lpf = get_cont_lf0(f0)
    cont_lf0_norm = (cont_lf0_lpf - mean) / std
    Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_norm)
    Wavelet_lf0_norm, _, _ = norm_scale(Wavelet_lf0)

    return Wavelet_lf0_norm


def get_lf0_cwt_norm(f0s, mean, std):
    uvs = []
    cont_lf0_lpfs = []
    cont_lf0_lpf_norms = []
    Wavelet_lf0s = []
    Wavelet_lf0s_norm = []
    scaless = []

    means = []
    stds = []
    for f0 in f0s:
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        cont_lf0_lpf_norm = (cont_lf0_lpf - mean) / std

        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)  # [560,10]
        Wavelet_lf0_norm, mean_scale, std_scale = norm_scale(Wavelet_lf0)  # [560,10],[1,10],[1,10]

        Wavelet_lf0s_norm.append(Wavelet_lf0_norm)
        uvs.append(uv)
        cont_lf0_lpfs.append(cont_lf0_lpf)
        cont_lf0_lpf_norms.append(cont_lf0_lpf_norm)
        Wavelet_lf0s.append(Wavelet_lf0)
        scaless.append(scales)
        means.append(mean_scale)
        stds.append(std_scale)

    return Wavelet_lf0s_norm, scaless, means, stds
"""这些函数主要用于对 LF0 序列进行连续小波变换（CWT）和归一化处理：

- `normalize_cwt_lf0`: 此函数接受一个 LF0 序列，并将其转换为连续 LF0，并将其归一化为与训练数据相同的标准化值。它首先使用 `get_cont_lf0` 函数将离散的声调序列转换为连续的 LF0 序列，然后将其标准化
为给定的均值和标准差，最后计算 LF0 序列的连续小波变换并对其进行归一化。

- `get_lf0_cwt_norm`: 此函数接受一个 LF0 序列列表，并对每个序列执行类似的操作。它首先将每个序列转换为连续 LF0，并将其归一化为给定的均值和标准差，然后计算每个序列的连续小波变换并对其进行归一化。最
后，它返回归一化的连续小波变换结果列表以及相应的尺度，均值和标准差。
"""


def inverse_cwt_torch(Wavelet_lf0, scales):
    import torch
    b = ((torch.arange(0, len(scales)).float().to(Wavelet_lf0.device)[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdim=True)) / lf0_rec_sum.std(-1, keepdim=True)
    return lf0_rec_sum


def inverse_cwt(Wavelet_lf0, scales):
    b = ((np.arange(0, len(scales))[None, None, :] + 1 + 2.5) ** (-2.5))
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdims=True)) / lf0_rec_sum.std(-1, keepdims=True)
    return lf0_rec_sum
"""
这两个函数用于将连续小波变换（CWT）的结果逆转回 LF0 序列：

inverse_cwt_torch: 此函数接受一个 PyTorch 张量 Wavelet_lf0 和一个尺度数组 scales，并返回逆连续小波变换的结果。它首先创建一个与输入张量相同的设备张量 b，然后对 Wavelet_lf0 进行一系列操作，
最终返回逆连续小波变换的结果。

inverse_cwt: 此函数接受一个 NumPy 数组 Wavelet_lf0 和一个尺度数组 scales，并返回逆连续小波变换的结果。它首先创建一个与输入数组相同形状的矩阵 b，然后对 Wavelet_lf0 进行一系列操作，最终返回
逆连续小波变换的结果
"""


def cwt2f0(cwt_spec, mean, std, cwt_scales):
    assert len(mean.shape) == 1 and len(std.shape) == 1 and len(cwt_spec.shape) == 3
    import torch
    if isinstance(cwt_spec, torch.Tensor):
        f0 = inverse_cwt_torch(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = f0.exp()  # [B, T]
    else:
        f0 = inverse_cwt(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = np.exp(f0)  # [B, T]
    return f0

def cwt2f0_norm(cwt_spec, mean, std, mel2ph, config):
    f0 = cwt2f0(cwt_spec, mean, std, config["cwt_scales"])
    f0 = torch.cat(
        [f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
    f0_norm = norm_f0(f0, None, config)
    return f0_norm
"""这段代码主要执行两个功能：

1. `cwt2f0`: 将连续小波变换（Continuous Wavelet Transform，CWT）的结果转换为基频（Fundamental Frequency，F0）。这个函数接收四个参数：`cwt_spec` 是连续小波变换的结果，`mean` 和 `std` 是
用于标准化的均值和标准差，`cwt_scales` 是小波尺度。函数首先检查输入的维度是否正确，然后根据输入的数据类型，使用不同的方式进行逆变换，最后进行反标准化操作，并将结果指数化，得到基频。

2. `cwt2f0_norm`: 在将连续小波变换结果转换为基频后，这个函数进一步将基频进行归一化处理。它接收了五个参数：`cwt_spec` 是连续小波变换的结果，`mean` 和 `std` 是用于标准化的均值和标准差，`mel2ph`
 是一个转换矩阵，用于从梅尔频谱到声相（Mel-spectrogram to phonetic）的转换，`config` 则是一些配置信息。该函数调用了 `cwt2f0` 函数来获得基频，然后根据输入的 `mel2ph` 矩阵，将基频延展到与其相
 同的长度，最后调用 `norm_f0` 函数对基频进行归一化处理，并返回归一化后的基频结果。
"""
