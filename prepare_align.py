import argparse

from utils.tools import get_configs_of
from preprocessor import ljspeech, vctk


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "VCTK" in config["dataset"]:
        vctk.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    config, *_ = get_configs_of(args.dataset)
    main(config)
    """这段代码的主要目的是根据命令行参数指定的数据集名称调用相应的准备对齐（prepare_align）函数，以准备数据集的对齐工作。根据数据集名称，它会调用适当的数据集准备函数，例如，如果数据集名称中
    包含"LJSpeech"，则会调用 LJSpeech 数据集的准备对齐函数（ljspeech.prepare_align），如果包含"VCTK"，则会调用 VCTK 数据集的准备对齐函数（vctk.prepare_align）。
    """
