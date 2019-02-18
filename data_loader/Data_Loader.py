import random
import numpy as np
import h5py
import scipy.ndimage


class DataLoader(object):

    def __init__(self, cfg):
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.train_data_dir = cfg.train_data_dir
        self.valid_data_dir = cfg.valid_data_dir
        self.test_data_dir = cfg.test_data_dir
        self.batch_size = cfg.batch_size
        self.num_tr = cfg.num_tr
        self.height, self.width = cfg.height, cfg.width
        self.max_bottom_left_front_corner = (cfg.dcfg.height*1 - 1, cfg.width*1 - 1)
        # maximum value that the bottom left front corner of a cropped patch can take

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            img_idx = np.sort(np.random.choice(self.num_tr, replace=False, size=self.batch_size))
            bottom = np.random.randint(self.max_bottom_left_front_corner[1])
            left = np.random.randint(self.max_bottom_left_front_corner[0])
            h5f = h5py.File(self.train_data_dir + 'train.h5', 'r')
            x = h5f['x_train'][img_idx, bottom:bottom + self.height, left:left + self.width, :]
            y = h5f['y_train'][img_idx, bottom:bottom + self.height, left:left + self.width]
            if self.augment:
                x, y = random_rotation_2d(x, y, max_angle=self.max_angle)
        elif mode == 'valid':
            h5f = h5py.File(self.valid_data_dir + 'valid.h5', 'r')
            x = h5f['x_valid'][start:end]
            y = h5f['y_valid'][start:end]
        elif mode == 'test':
            h5f = h5py.File(self.test_data_dir + 'test1.h5', 'r')
            x = h5f['x_test'][start:end]
            y = h5f['y_test'][start:end]
        h5f.close()
        return x, y

    def count_num_samples(self, mode='valid'):
        if mode == 'valid':
            h5f = h5py.File(self.valid_data_dir + 'valid.h5', 'r')
            num_ = h5f['y_valid'][:].shape[0]
        elif mode == 'test':
            h5f = h5py.File(self.test_data_dir + 'test1.h5', 'r')
            num_ = h5f['y_test'][:].shape[0]
        h5f.close()
        return num_


def random_rotation_2d(img_batch, mask_batch, max_angle):
    """
    Randomly rotate an image by a random angle (-max_angle, max_angle)
    :param img_batch: batch of 3D images
    :param mask_batch: batch of 3D masks
    :param max_angle: `float`. The maximum rotation angle
    :return: batch of rotated 3D images and masks
    """
    size = img_batch.shape
    img_batch = np.squeeze(img_batch, axis=-1)
    img_batch_rot, mask_batch_rot = img_batch, mask_batch
    for i in range(img_batch.shape[0]):
        axis_rot = np.random.randint(2)
        if axis_rot:   # if rotating along any axis
            image, mask = img_batch[i], mask_batch[i]
            # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            img_batch_rot[i] = rotate(image, angle)
            mask_batch_rot[i] = rotate(mask, angle)

    return img_batch_rot.reshape(size), mask_batch


def rotate(x, angle):
    return scipy.ndimage.interpolation.rotate(x, angle, mode='nearest', axes=(0, 1), reshape=False)
