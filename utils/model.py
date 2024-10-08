import os
import json

import torch
import numpy as np

import hifigan
from model import DiffGANTTS, JCUDiscriminator, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    epoch = 1
    model = DiffGANTTS(args, preprocess_config, model_config, train_config).to(device)
    discriminator = JCUDiscriminator(preprocess_config, model_config, train_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        epoch = int(ckpt["epoch"])
        model.load_state_dict(ckpt["G"]) # named_parameters: {'variance_adaptor', 'diffusion', 'mel_linear', 'text_encoder', 'decoder'}
        discriminator.load_state_dict(ckpt["D"]) # named_parameters: {'input_projection', 'cond_conv_block', 'uncond_conv_block', 'conv_block', 'mlp'}

        # if args.model == "shallow": # this is replaced by detaching all input/output of aux model
        #     freeze_model(model, except_named=["diffusion"])
        """这个函数用于获取模型对象，根据参数中的配置信息和设备，初始化并加载模型的权重。主要功能包括：

- 实例化模型对象和鉴别器对象，这些对象在训练时会在给定设备上进行运算。
- 如果指定了 `restore_step`，则加载该步骤的检查点，即模型和鉴别器的权重。
- 返回加载的模型对象。

在加载检查点时，它执行以下操作：
- 构建检查点文件的路径。
- 使用 `torch.load` 加载检查点文件。
- 从检查点中提取模型和鉴别器的状态字典（即权重），并将其加载到相应的模型和鉴别器对象中。
- 如果需要，还可以在这里冻结模型的特定部分，但当前的代码注释掉了这部分。

最终，该函数返回加载的模型对象。
"""

    if train:
        init_lr_G = train_config["optimizer"]["init_lr_G"]
        init_lr_D = train_config["optimizer"]["init_lr_D"]
        betas = train_config["optimizer"]["betas"]
        gamma = train_config["optimizer"]["gamma"]
        optG_fs2 = ScheduledOptim(model, train_config, model_config, args.restore_step)
        optG = torch.optim.Adam(model.parameters(), lr=init_lr_G, betas=betas)
        optD = torch.optim.Adam(discriminator.parameters(), lr=init_lr_D, betas=betas)
        sdlG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma)
        sdlD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma)
        if args.restore_step and args.restore_step != train_config["step"]["total_step_aux"]: # should be initialized when "shallow"
            optG_fs2.load_state_dict(ckpt["optG_fs2"])
            optG.load_state_dict(ckpt["optG"])
            optD.load_state_dict(ckpt["optD"])
            sdlG.load_state_dict(ckpt["sdlG"])
            sdlD.load_state_dict(ckpt["sdlD"])
        model.train()
        discriminator.train()
        return model, discriminator, optG_fs2, optG, optD, sdlG, sdlD, epoch

    model.eval()
    model.requires_grad_ = False
    return model
"""这部分代码根据是否处于训练模式来初始化模型和鉴别器的优化器，并返回相应的对象。主要功能包括：

- 如果是训练模式：
  - 从训练配置中获取优化器的初始学习率、动量参数和衰减率等参数。
  - 根据模型的类型（`DiffGANTTS`），初始化优化器。这里使用了 `ScheduledOptim` 类来调度学习率的变化，以及 `torch.optim.Adam` 类来优化模型参数。
  - 如果指定了 `restore_step`，则加载训练中断时保存的优化器状态，即学习率和动量等参数。
  - 将模型和鉴别器设置为训练模式。
  - 返回模型、鉴别器和优化器等对象，以及当前的训练轮次。

- 如果是推理模式：
  - 将模型设置为评估模式，并关闭梯度追踪。
  - 返回模型对象。

在训练模式下，函数返回了许多对象，包括模型、鉴别器、两个优化器、两个学习率调度器以及当前的训练轮次。这些对象将用于训练循环中的优化步骤。在推理模式下，函数只返回了模型对象，因为此时不需要进行优化。
"""


# def freeze_model(model, except_named=[""]):
#     for name, p in model.named_parameters():
#         p.requires_grad = True if name in except_named else False


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_netG_params(model_kernel):
    return list(model_kernel.C.parameters()) \
        + list(model_kernel.Z.parameters()) \
        + list(model_kernel.G.parameters())


def get_netD_params(model_kernel):
    return model_kernel.D.parameters()


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder
"""这段代码定义了一些辅助函数，用于获取模型参数数量、提取生成器和鉴别器的参数，以及加载和配置声码器模型。让我们逐个解释：

1. `get_param_num(model)`: 这个函数接受一个模型对象并返回其参数数量。

2. `get_netG_params(model_kernel)`: 这个函数接受一个生成器模型对象，并返回生成器的参数列表。在这里，假设生成器模型有三个部分：`C`、`Z`、`G`。

3. `get_netD_params(model_kernel)`: 这个函数接受一个鉴别器模型对象，并返回鉴别器的参数列表。在这里，假设鉴别器模型只有一个部分：`D`。

4. `get_vocoder(config, device)`: 这个函数根据配置加载和配置声码器模型。根据配置的不同，可以选择加载不同的声码器模型。支持的声码器模型包括 MelGAN 和 HiFi-GAN。MelGAN 是一个生成式对抗网络，用
于语音合成，而 HiFi-GAN 是一个高保真度的声码器模型。

   - 如果选择 MelGAN，根据配置中的发音人信息加载预训练的 MelGAN 模型。
   - 如果选择 HiFi-GAN，根据配置中的发音人信息加载预训练的 HiFi-GAN 模型。

以上这些函数可以在模型训练和推断过程中使用，以方便地获取模型参数和加载声码器模型。
"""


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
"""这段代码定义了一个函数`vocoder_infer`，用于推断声码器模型生成音频。让我们来解释一下这个函数：

- `vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None)`: 这个函数接受如下参数：
  - `mels`: 包含多个梅尔频谱的列表或数组，用于声码器模型的推断。
  - `vocoder`: 声码器模型对象，用于将梅尔频谱转换为音频。
  - `model_config`: 包含声码器模型配置的字典，其中可能包含声码器的类型。
  - `preprocess_config`: 包含预处理配置的字典，用于音频的后处理。
  - `lengths`: 可选参数，包含每个梅尔频谱对应的长度。

在函数内部，首先根据声码器的类型选择合适的声码器模型进行推断：

- 如果声码器类型是 "MelGAN"，则通过将梅尔频谱除以自然对数 10，然后将其输入到 MelGAN 的逆转换函数中，得到音频。
- 如果声码器类型是 "HiFi-GAN"，则将梅尔频谱输入到 HiFi-GAN 模型中，得到音频。

接着，将生成的音频转换为整型数据，并根据预处理配置的最大音频值对其进行缩放。

最后，如果提供了长度信息，则对每个生成的音频进行截断，使其长度与相应的梅尔频谱匹配。

函数返回生成的音频列表。
"""
