import json
from pathlib import Path

def load_args(file_dir:str,file_name:str) -> dict:
    """
    Load arguments from a JSON file.

    Args:
    - file_dir (str): Directory where the JSON file is located.
    - file_name (str): Name of the JSON file.

    Returns:
    - dict: Dictionary containing the loaded arguments.
    """
    json_file = Path(file_dir,file_name)
    
    with open(json_file,"r") as file:
        args = json.loads(file.read())
        
    return args
    