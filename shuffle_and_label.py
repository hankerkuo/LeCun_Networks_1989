import numpy as np
import os
import shutil
import imageio
from six.moves import cPickle as pickle

image_size = 16  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
# the functions here, we need to put '/' in the end of each path


def one_hot(y_value, num_class):
    # delicate index operation, y_value is a 1D array, num_class is the amount of class (i.e for MNIST, num_class = 10)
    # reference: https://stackoverflow.com/questions/29831489/numpy-1-hot-array
    hot = np.zeros((len(y_value), num_class))
    hot[np.arange(len(y_value)), y_value] = 1
    return hot


def random_take_data(mother_folder, train_num):                                     # put train and test data into corresponding folder
                                                                                    # be sure the mother_folder has '/' in the end
    if not os.path.exists('./train_data'):                                          # create folders
        os.makedirs('./train_data')
    if not os.path.exists('./test_data'):
        os.makedirs('./test_data')

    for kid_folder in os.listdir(mother_folder):                                    # go through each class
        image_files = os.listdir(mother_folder + kid_folder)                        # read all the files in a kid folder (e.g. each class)
        image_files_train = np.random.choice(image_files, train_num, replace=False) # randomly choosing train data
        image_files_test = np.setdiff1d(image_files, image_files_train)             # the remaining data becomes test data
        for image in image_files_train:                                             # copy the images to new folders
            shutil.copy(mother_folder + kid_folder + '/' + image, './train_data')
        for image in image_files_test:
            shutil.copy(mother_folder + kid_folder + '/' + image, './test_data')


def shuffle_and_label_normalize(folder):                                            # return data feature(in array) and label after shuffle
    image_files = os.listdir(folder)
    np.random.shuffle(image_files)

    # making new arrays for storing data and lalel
    data_set = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    label_set = np.ndarray(shape=(len(image_files)), dtype=np.int)

    image_amount = 0
    for image in image_files:
        data_set[image_amount, :, :] = (np.sum(imageio.imread(folder + image).astype(float), axis=2) / 3 -
                                        pixel_depth / 2) / pixel_depth              # normalize the data, because the image is RGB
        label_set[image_amount] = image[0]                                          # the first letter of the file name is it's class
        image_amount = image_amount + 1
    label_set = one_hot(label_set, 10)                                              # making the label set 'one-hot'

    # creat pickle files (for each data and label)
    try:
        with open(folder + 'data.pickle', 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
        with open(folder + 'label.pickle', 'wb') as f:
            pickle.dump(label_set, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', folder + 'label.pickle', 'wb', ':', e)


def shuffle_and_label_binary(folder):
    image_files = os.listdir(folder)
    np.random.shuffle(image_files)
    data_set = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    label_set = np.ndarray(shape=(len(image_files)), dtype=np.int)
    image_amount = 0
    for image in image_files:
        # this part makes the entries of image matrix be only 1 and 0
        # because the image is RGB, but only black and white appear, so each channel's entry is either 255 or 0
        # use np.sum at axis=2 -> value will be either 765.0 ( 255.0 * 3) or 0
        # so by dividing 765, we can get binary values
        data_set[image_amount, :, :] = np.sum(imageio.imread(folder + image).astype(float), axis=2) / 765.0
        label_set[image_amount] = image[0]
        image_amount = image_amount + 1
    label_set = one_hot(label_set, 10)
    try:
        with open(folder + 'data.pickle', 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
        with open(folder + 'label.pickle', 'wb') as f:
            pickle.dump(label_set, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', folder + 'label.pickle', 'wb', ':', e)

# this part is the pre_processing part

# random_take_data('C:/data/digits/', 32)
# shuffle_and_label_binary('./train_data/')

# this part is the pre_processing part


