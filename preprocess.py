import argparse

from utils.tools import get_configs_of
from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    preprocessor = Preprocessor(preprocess_config, model_config, train_config)
    preprocessor.build_from_path()
    """这段代码的主要作用是使用命令行参数指定的数据集名称，加载该数据集的配置信息，并使用加载的配置信息构建预处理器（Preprocessor）对象。然后，调用 `build_from_path()` 方法开始构建预处理器，该
    方法可能会从文件路径中构建预处理器所需的数据结构或执行其他必要的操作。
    """
