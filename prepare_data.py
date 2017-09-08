
import os
import cv2
import h5py
import numpy


def prepare_training_data():
    '''
    module to prepare hdf5 train data with data and label pairs
    '''
    names = os.listdir(DATA_PATH)
    names = sorted(names)
    nums = names.__len__()

    data = numpy.zeros((train_img_num * random_crop, 1,
                        patch_size, patch_size), dtype=numpy.double)
    label = numpy.zeros((train_img_num * random_crop, 1,
                         label_size, label_size), dtype=numpy.double)

    if nums < train_img_num:
        print("training img is not enough")

    for i in range(train_img_num):
        name = DATA_PATH + names[i]
        print ("TRAINING: image - ", name)
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(
            hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # produce random_crop random coordinate to crop training img
        Points_x = numpy.random.randint(
            0, min(shape[0], shape[1]) - patch_size, random_crop)
        Points_y = numpy.random.randint(
            0, min(shape[0], shape[1]) - patch_size, random_crop)

        for j in range(random_crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] +
                              patch_size, Points_y[j]: Points_y[j] + patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] +
                              patch_size, Points_y[j]: Points_y[j] + patch_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * random_crop + j, 0, :, :] = lr_patch
            label[i * random_crop + j, 0, :, :] = hr_patch[conv_side: -
                                                           conv_side, conv_side: -conv_side]
            # cv2.imshow("lr", lr_patch)
            # cv2.imshow("hr", hr_patch)
            # cv2.waitKey(0)
    return data, label


def prepare_testing_data():
    '''
    module to prepare hdf5 test data with data and label pairs
    '''
    names = os.listdir(TEST_PATH)
    names = sorted(names)
    nums = names.__len__()

    data = numpy.zeros((test_img_num * random_crop, 1,
                        patch_size, patch_size), dtype=numpy.double)
    label = numpy.zeros((test_img_num * random_crop, 1,
                         label_size, label_size), dtype=numpy.double)

    if nums < test_img_num:
        print("training img is not enough")

    for i in range(test_img_num):
        name = TEST_PATH + names[i]
        print ("TESTING: image - ", name)
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(
            hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # produce random_crop random coordinate to crop training img
        Points_x = numpy.random.randint(
            0, min(shape[0], shape[1]) - patch_size, random_crop)
        Points_y = numpy.random.randint(
            0, min(shape[0], shape[1]) - patch_size, random_crop)

        for j in range(random_crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] +
                              patch_size, Points_y[j]: Points_y[j] + patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] +
                              patch_size, Points_y[j]: Points_y[j] + patch_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * random_crop + j, 0, :, :] = lr_patch
            label[i * random_crop + j, 0, :, :] = hr_patch[conv_side: -
                                                           conv_side, conv_side: -conv_side]
            # cv2.imshow("lr", lr_patch)
            # cv2.imshow("hr", hr_patch)
            # cv2.waitKey(0)
    return data, label


def write_hdf5(data, labels, output_filename):
    """
    write and save image data and its label(s) to hdf5 file.
    contain data and label
    """
    x = data.astype(numpy.float32)
    y = labels.astype(numpy.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


def read_training_data(file):
    '''
    to read the data from hdf5 files containing data and label
    '''
    with h5py.File(file, 'r') as hf:
        data = numpy.array(hf.get('data'))
        label = numpy.array(hf.get('label'))
        train_data = numpy.transpose(data, (0, 2, 3, 1))
        train_label = numpy.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


def set_variables():
    '''
    set all the configurations here
    '''
    if not os.path.exists('./model'):
        os.makedirs('./model')
    from configparser import SafeConfigParser
    global DATA_PATH, TEST_PATH, train_img_num, test_img_num
    global random_crop, patch_size, label_size, conv_side, scale
    config = SafeConfigParser()
    config.read('config.ini')
    DATA_PATH = config.get('main', 'data_path')
    TEST_PATH = config.get('main', 'test_path')
    train_img_num = config.getint('main', 'train_img_num')
    test_img_num = config.getint('main', 'test_img_num')
    random_crop = config.getint('main', 'random_crop')
    patch_size = config.getint('main', 'patch_size')
    label_size = config.getint('main', 'label_size')
    conv_side = config.getint('main', 'conv_side')
    scale = config.getint('main', 'scale')


def main():
    '''
    entry point calls
    '''
    set_variables()
    data, label = prepare_training_data()
    write_hdf5(data, label, "./model/train.h5")
    data, label = prepare_testing_data()
    write_hdf5(data, label, "./model/test.h5")
    print("Data preparation completed.")
    # _, _a = read_training_data("train.h5")
    # _, _a = read_training_data("test.h5")

if __name__ == "__main__":
    '''
    entry point of the program
    '''
    main()