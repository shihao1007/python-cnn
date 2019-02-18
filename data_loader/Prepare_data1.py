import os
from PIL import Image
import numpy as np
import h5py


def rgb2ind(rgb, num=3):
    new = np.zeros((rgb.shape[0], rgb.shape[1], num))
    im_size=(rgb.shape[0], rgb.shape[1])
    r = rgb[:,:,0]
    r[np.where(r<100)]=0
    g = rgb[:, :, 1]
    g[np.where(g < 100)] = 0
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    for ch in range(num-1):
        new[:,:,ch+1] = rgb[:,:,ch]
    ind0 = np.ravel_multi_index(np.where(rgb[:, :, 0] == 0), im_size)
    ind1 = np.ravel_multi_index(np.where(rgb[:, :, 1] == 0), im_size)
    inter1 = list(set(ind0).intersection(set(ind1)))
    bg = np.zeros(im_size)
    bg[np.unravel_index(inter1, im_size)] = 255
    new[:, :, 0] = bg
    return(np.argmax(new, axis=2))



def read_images(path, img_size=(128, 128), num_cls=2, label=False):
    """
    reads all images in a folder one-by-one (in alphabetical order) and converts them to 3D images
    :param path: folder path
    :param img_size: size of final 2D images
    :return: 2D image of size (1, height, width, 1)
    """
    img = np.array(Image.open(path))
    if (label and num_cls>2):
        img = rgb2ind(img, num_cls)
    elif len(img.shape) == 3:     # if image is saved as RGB (i.e. with three similar channels)
        img = img[:, :, 0]
    return np.expand_dims(img, axis=0)

def get_data(path, num_cls=2, img_size=(128, 128)):
    """
    loads all images and corresponding masks and converts them to numpy arrays
    :param path: path to the destination folder (which includes raw and mask folders)
    :param img_size: size of the images
    :return: 4D arrays (#images, height, width, depth) of images and corresponding masks
    """
    folders = os.listdir(path)
    folders.sort()
    annotation_folder = folders[:int(len(folders) / 2)]
    input_folder = folders[int(len(folders) / 2):]
    images = np.zeros([0] + list(img_size))
    masks = np.zeros([0] + list(img_size))
    imgs = [n for n in os.listdir(os.path.join(path, input_folder[0]))]
    for img in imgs:
        img_path = os.path.join(path, input_folder[0], img)
        image = read_images(img_path, img_size)
        images = np.concatenate((images, image), axis=0)
    images = images/255
    msks = [m for m in os.listdir(os.path.join(path, annotation_folder[0]))]
    for msk in msks:
        msk_path =os.path.join(path, annotation_folder[0], msk)
        mask = read_images(msk_path, img_size, num_cls, label=True)
        masks = np.concatenate((masks, mask), axis=0)
    masks = (masks).astype(int)
    return images, masks


def normalize(x):
    """
    Normalizes the input to have zero mean and unit standard deviation
    :param x: input of size (#images, height, width, depth)
    :return: normalized input of size (#images, height, width, depth, 1), mean and std arrays of all images
    """
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    x_norm = (x-m)/s
    return np.expand_dims(x_norm, axis=-1), m, s

num_cls = 3

# Create normalize, and save training data
train_path = './data/KESM_data/train_data/'
x_train, y_train = get_data(train_path, num_cls, img_size=(128, 128))
#x_train, m_train, s_train = normalize(x_train)
x_train = np.expand_dims((x_train), axis=-1)
h5f = h5py.File(train_path + 'train.h5', 'w')
h5f.create_dataset('x_train', data=x_train)
h5f.create_dataset('y_train', data=y_train)
#h5f.create_dataset('m_train', data=m_train)
#h5f.create_dataset('s_train', data=s_train)
h5f.close()

# h5f = h5py.File(train_path + 'train.h5', 'r')
# m_train = h5f['m_train'][:]
# s_train = h5f['s_train'][:]
# h5f.close()

# Create normalize, and save validation data
valid_path = './data/KESM_data/valid_data/'
x_valid, y_valid = get_data(valid_path, num_cls, img_size=(64, 64))

# neither the best validation set, nor the best way to normalize
#x_valid = np.expand_dims((x_valid-m_train[:64, :64, :32])/s_train[:64, :64, :32], axis=-1)
x_valid = np.expand_dims((x_valid), axis=-1)
h5f = h5py.File(valid_path + 'valid.h5', 'w')
h5f.create_dataset('x_valid', data=x_valid)
h5f.create_dataset('y_valid', data=y_valid)
h5f.close()

# Create normalize, and save test data
test_path = './data/KESM_data/test1/'
x_test, y_test = get_data(test_path, num_cls, img_size=(64, 64))

x_test = np.expand_dims((x_test), axis=-1)
h5f = h5py.File(test_path + 'test.h5', 'w')
h5f.create_dataset('x_test', data=x_test)
h5f.create_dataset('y_test', data=y_test)
h5f.close()
