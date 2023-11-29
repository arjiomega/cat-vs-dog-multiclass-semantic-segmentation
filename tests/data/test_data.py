import numpy as np
from src.data.data_utils import load_raw_data, load_specie_dict, remove_unlabeled, preprocess



class TestLoadData:
    @classmethod
    def setup_class(cls):
        """Called before every class initialization."""
        cls.img_list, cls.mask_list = load_raw_data.load()
        cls.specie_dict = load_specie_dict.load()
        cls.img_list, cls.mask_list = remove_unlabeled.remove(cls.img_list, cls.mask_list,cls.specie_dict) # type: ignore

    def test_length(self):
        if_false = "length of image and mask list must be the same"
        assert len(self.img_list) == len(self.mask_list), if_false

    def test_name_equality(self):
        if_false = "all image-mask pair must have the same indices"
        assert all(img.split(".")[0] == mask.split(".")[0] \
                   for (img,mask) in zip(self.img_list,self.mask_list)), if_false
        
    def test_label_length(self):
        if_false = "labels and images must have the same length"
        assert len(self.specie_dict) == len(self.img_list), if_false

    @classmethod
    def teardown_class(cls):
        """Called after every class initialization."""
        del cls.img_list, cls.mask_list, cls.specie_dict
            
class TestPreprocess:
    @classmethod
    def setup_class(cls):
        """Called before every class initialization."""
        pass

    def test_load(self):
        pass

    def test_fix(self):
        test_label = 1 # cat
        test_input = np.array([[1,2,3],[3,3,3],[2,2,1]])

        test_preprocess = preprocess.Preprocess("test_file",test_label)

        test_fix_mask = test_preprocess.fix(test_input)

        expected_output = np.array([[1,0,1],[1,1,1],[0,0,1]])

        print(test_fix_mask)
        print(expected_output)

        assert (test_fix_mask == expected_output).all(), "fix fail"

    def test_save(self):
        pass
