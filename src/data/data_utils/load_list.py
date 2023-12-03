import os

def load(img_dir:str,mask_dir:str) -> list[str]:
    """
    Loads image and mask filenames from specified directories, performs integrity checks, and returns a list of data.

    Args:
    - img_dir (str): Directory path containing image files.
    - mask_dir (str): Directory path containing mask files.

    Returns:
    - data_list (list): A list of filenames without file extensions, representing the loaded data.
    """
    img_list = [img_ for img_ in os.listdir(img_dir) if not img_.startswith(".") and (img_.endswith(".jpg") or img_.endswith(".png"))]
    mask_list =[mask_ for mask_ in os.listdir(mask_dir) if not mask_.startswith(".") and (mask_.endswith(".jpg") or mask_.endswith(".png"))]
    img_list = sorted(img_list)
    mask_list = sorted(mask_list)
    
    img_mask_pair_similarity_test(img_list,mask_list) # consider removing this since data already tested
    
    data_list = remove_ext(img_list)
    
    return data_list

def img_mask_pair_similarity_test(img_list:list[str],mask_list:list[str]):
    """
    Performs a similarity check between image and mask filenames to ensure matching pairs.

    Args:
    - img_list (list): List of image filenames.
    - mask_list (list): List of mask filenames.

    Raises:
    - AssertionError: If the filenames without extensions in img_list and mask_list don't match.
    """
    
    img_list_no_ext = remove_ext(img_list)
    mask_list_no_ext = remove_ext(mask_list)
    
    assert img_list_no_ext == mask_list_no_ext, "img_list and mask_list are not the same!"
    
def remove_ext(file_list:list[str]) -> list[str]:
    """
    Removes file extensions from a list of filenames.

    Args:
    - file_list (list): List of filenames.

    Returns:
    - list: List of filenames without file extensions.
    """
    
    file_list_no_ext = [file.split(".")[0] for file in file_list]
    
    return file_list_no_ext