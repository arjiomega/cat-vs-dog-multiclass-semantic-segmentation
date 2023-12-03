""" module description """

from pathlib import Path
import pandas as pd

def load(file_dir:str,file_name:str) -> dict[str,int]:
    """
    Loads data from a CSV file containing image information and returns a dictionary mapping image names to species indices.

    Args:
    - file_dir (str): Directory path where the CSV file is located.
    - file_name (str): Name of the CSV file.

    Returns:
    - specie_dict (Dict[str, int]): A dictionary containing image names as keys and corresponding species indices as values.
    """

    df = pd.read_csv(Path(file_dir,file_name), comment='#',header=None, sep=" ")
    df.columns = ['image_name','class_index','specie_index','breed_index']
    df = df.sort_values(by = 'image_name', ascending=True)
    df = df.reset_index(drop=True)

    # Get dict ("image_name": specie_index)
    specie_dict = df.set_index("image_name")["specie_index"].to_dict()

    return specie_dict