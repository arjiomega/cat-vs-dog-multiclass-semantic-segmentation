import numpy as np
import tensorflow as tf


class BatchLoader(tf.keras.utils.Sequence):
    """
    Generates batches of data for a Keras model using a custom dataset.

    Args:
    - dataset (CustomDataset): Custom dataset containing image paths or other data.
    - batch_size (int): Size of each batch.
    - shuffle (bool, optional): Flag indicating whether to shuffle the dataset between epochs. Default is False.

    Methods:
    - __len__(): Returns the number of batches per epoch.
    - __getitem__(i): Generates a batch of data at index i.
    - on_epoch_end(): Shuffles the dataset indexes at the end of each epoch if shuffle is enabled.
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False):

        self.dataset = dataset
        self.dataset_size = len(dataset.img_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(self.dataset_size)

        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.

        Returns:
        - int: Number of batches per epoch.
        """
        
        # Number of Batches per Epoch
        return int(np.floor(self.dataset_size / self.batch_size))  # len(self.list_IDs)

    def __getitem__(self,i:int) -> list[np.ndarray,np.ndarray]:
        """
        Generates a batch of data at index i.

        Args:
        - i (int): Index of the batch.

        Returns:
        - list: Batch of data.
        """
        
        start = i * self.batch_size
        stop = (i+1) * self.batch_size
        data_indices = self.indices[start:stop]
        data = [self.dataset[j] for j in data_indices]

        # data = [(img1,mask1),(img2,mask2),...]
        # batch = [stack of images, stack of masks] 
        # stack of images shape = (num_batches,width,height,num_channels)
        # stack of masks shape = (num_batches,width,height,num_classes)
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def on_epoch_end(self):
        """
        Shuffles the dataset indexes at the end of each epoch if shuffle is enabled.
        """
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)