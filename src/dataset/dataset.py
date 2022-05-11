import tensorflow as tf
from PIL import Image
from tensorflow import keras

class BaseDataset(keras.utils.Sequence):
    def __init__(self, img_raw_paths, img_mask_paths,label_paths, batch_size):
        """The constructor of the dataset, x_set, y_set, z_set should be mapped to the same index
            For example when access to x_set[0], y_set[0] and z_set[0], we can get the 
            "img_01_raw.jpg, img_01_mask.png, img_01_label.json"
            
        Args:
            x_set (list): list of raw image paths
            y_set (list): list of mask image paths
            z_set (list): list of label paths
            batch_size (int): the batch size for each iteration
        """
        self.img_raw_paths = img_raw_paths
        self.img_mask_paths = img_mask_paths
        self.label_paths = label_paths
        self.batch_size = batch_size

    def __len__(self):
        return tf.math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # get candidate indices
        batch_imgs_raw = self.x[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_imgs_mask = self.y[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_labels = self.z[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        # read imgs and convert it to tensor
        batch_img_raw_train = tf.convert_to_tensor([Image.open(file_name).resize(640,640) for file_name in batch_imgs_raw], dtype=tf.float32)
        batch_img_mask_train = tf.convert_to_tensor([Image.open(file_name).resize(640,640) for file_name in batch_imgs_mask], dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
        return batch_img_raw_train, batch_img_mask_train, batch_labels
