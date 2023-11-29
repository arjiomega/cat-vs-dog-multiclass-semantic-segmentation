from src.data.data_utils import load_raw_data, load_specie_dict, remove_unlabeled, preprocess

if __name__ == '__main__':

    img_list, mask_list = load_raw_data.load()
    specie_dict = load_specie_dict.load()
    img_list, mask_list = remove_unlabeled.remove(img_list,mask_list,specie_dict) # type: ignore

    # saving
    run = preprocess.Run(img_list=img_list,
                   mask_list=mask_list,
                   specie_dict=specie_dict)
    run.save_preprocessed_data()