import os

import pandas as pd

from config import config

def load():

    df = pd.read_csv((os.path.join(config.RAW_DATA_DIR,'annotations','list.txt')), comment='#',header=None, sep=" ")
    df.columns = ['image_name','class_index','specie_index','breed_index']
    df = df.sort_values(by = 'image_name', ascending=True)
    df = df.reset_index(drop=True)

    # Get dict ("image_name": specie_index)
    specie_dict = df.set_index("image_name")["specie_index"].to_dict()

    return specie_dict