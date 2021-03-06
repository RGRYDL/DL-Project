"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division 
import math
import pprint
import scipy.misc
import numpy as np
import copy
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img.astype(np.uint8), [fine_size, fine_size])
    img = img.astype(np.float)
    img = img/127.5 - 1
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1], is_grayscale = True)
    img_B = img_B[:, :, np.newaxis]
    img_B = np.concatenate((img_B, img_B, img_B), axis=2)
    img_C = imread(image_path[2])

    if not is_testing:
        img_A = scipy.misc.imresize(img_A.astype(np.uint8), [load_size, load_size])
        img_B = scipy.misc.imresize(img_B.astype(np.uint8), [load_size, load_size])
        img_C = scipy.misc.imresize(img_C.astype(np.uint8), [load_size, load_size])

        #h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        #w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        #img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        #img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]
        #img_C = img_C[h1:h1+fine_size, w1:w1+fine_size]

        #if np.random.random() > 0.5:
            #img_A = np.fliplr(img_A)
            #img_B = np.fliplr(img_B)
            #img_C = np.fliplr(img_C)
    else:
        img_A = scipy.misc.imresize(img_A.astype(np.uint8), [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B.astype(np.uint8), [fine_size, fine_size])
        img_C = scipy.misc.imresize(img_C.astype(np.uint8), [fine_size, fine_size])

    img_A = img_A.astype(np.float)
    img_B = img_B.astype(np.float)
    img_C = img_C.astype(np.float)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_C = img_C/127.5 - 1.

    img_B = img_B[:, :, 0]
    img_B = img_B[:, :, np.newaxis]

    img_ABC = np.concatenate((img_A, img_B, img_C), axis=2)
    # img_ABC shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ABC

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size).astype(np.uint8))

def inverse_transform(images):
    return (images+1.)*127.5
