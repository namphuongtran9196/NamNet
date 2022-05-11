import tensorflow as tf
from PIL import Image
from tensorflow import keras

class BaseDataset(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return tf.math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *self.batch_size]

        batch_x_train = tf.convert_to_tensor([Image.open(file_name).resize(640,640) for file_name in batch_x], dtype=tf.float32)
        batch_y_train = tf.convert_to_tensor(batch_y, dtype=tf.float32)
        return batch_x_train, batch_y_train
