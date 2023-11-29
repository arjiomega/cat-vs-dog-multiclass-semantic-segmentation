

# Since specie_dict and img_list do not have the same length, we now check which of the images do not have label
def remove(img_list,mask_list,specie_dict,show_log=False,get_unlabeled=False):

    no_labels = []
    upd_img_list = []
    upd_mask_list = []
    for img,mask in zip(img_list,mask_list):

        if img.split(".")[0] not in specie_dict:
            no_labels.append(img)
            if show_log:
                print(f"{img} has no label")
        else: 
            upd_img_list.append(img)
            upd_mask_list.append(mask)

    if show_log:
        print(f"count of images without label: {len(no_labels)}")

    if not get_unlabeled:
        return(upd_img_list, upd_mask_list)  
    else:
        return(upd_img_list, upd_mask_list, no_labels)
        
