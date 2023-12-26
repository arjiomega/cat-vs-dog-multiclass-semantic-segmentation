import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
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


if __name__ == "__main__":
    # load base data
    base_loader = loadbasedata.LoadBaseData(
        labelpath=config.RAW_DATA_DIR,
        labelfilename=Path("annotations", "list.txt"),
        show_logs=True,
    )
    train_set, valid_set, test_set = base_loader.load_datasets()

    multiprocess_save_basedata(
        base_loader, train_set, valid_set, test_set, savepath=config.PROCESSED_DATA_DIR
    )

    # load data from label data
    gen_custom_data = gencustomdata.GenCustomData(
        json_path=Path(config.NEW_DATA_DIR, "annotations.json"),
        savepath=Path(config.PROCESSED_DATA_DIR, "debugging_set"),
    )
    gen_custom_data.save()
