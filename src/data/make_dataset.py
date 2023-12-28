import os
import shutil
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from src.utils.load_args import load_args
from src.data.generate import loadbasedata, gencustomdata


def multiprocess_save_basedata(
    base_loader: loadbasedata.LoadBaseData,
    train_set: pd.DataFrame,
    valid_set: pd.DataFrame,
    test_set: pd.DataFrame,
    savepath: Path,
):
    processor_count = multiprocessing.cpu_count()

    train_split = np.array_split(train_set, processor_count)
    valid_split = np.array_split(valid_set, processor_count)
    test_split = np.array_split(test_set, processor_count)

    processes = []

    for i in range(processor_count):
        p = multiprocessing.Process(
            target=base_loader.save_basedata,
            args=(train_split[i], valid_split[i], test_split[i], savepath),
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def load_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_args",
        dest="train_args",
        type=str,
        help="json filename in config dir containing the training setup. Ex. 'setup1.json'",
    )
    input_args = parser.parse_args()

    if not input_args.train_args:
        parser.error("training setup file is required.")

    return input_args


if __name__ == "__main__":
    # Load CLI args
    input_args = load_input_args()

    # Training setup
    args = load_args(config.CONFIG_DIR, input_args.train_args)

    # Remove processed data dir if exists
    if os.path.exists(config.PROCESSED_DATA_DIR):
        shutil.rmtree(config.PROCESSED_DATA_DIR)

    # load base data
    base_loader = loadbasedata.LoadBaseData(
        labelpath=config.RAW_DATA_DIR,
        labelfilename=Path("annotations", "list.txt"),
        show_logs=True,
    )
    train_set, valid_set, test_set = base_loader.load_datasets(
        datasplit=(
            args["setup"]["train_split"],
            args["setup"]["valid_split"],
            args["setup"]["test_split"],
        )
    )

    multiprocess_save_basedata(
        base_loader, train_set, valid_set, test_set, savepath=config.PROCESSED_DATA_DIR
    )

    # load data from label data
    gen_custom_data = gencustomdata.GenCustomData(
        json_path=Path(config.NEW_DATA_DIR, "annotations.json"),
        savepath=Path(config.PROCESSED_DATA_DIR, "debugging_set"),
    )
    gen_custom_data.save()
