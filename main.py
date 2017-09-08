
import sys
import getopt
import numpy
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import adam
import prepare_data as pd
import psnr
import cv2


def model():
    '''
    Creates a model of the training exexution
    '''
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='he_normal',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(1, 1), kernel_initializer='he_normal',
                     activation='relu', padding='valid', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5, 5), kernel_initializer='he_normal',
                     activation='linear', padding='valid', use_bias=True))
    Adam = adam(lr=0.001)
    SRCNN.compile(optimizer=Adam, loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return SRCNN


def train():
    '''
    Train the model, need train.h5 -> run prepare_data.py 
    before training module execution
    '''
    pd.main()
    srcnn_model = model()
    data, label = pd.read_training_data("./model/train.h5")
    # srcnn_model.load_weights("m_model_adam.h5")
    srcnn_model.fit(data, label, batch_size=128, epochs=30)
    srcnn_model.save_weights("./model/srcnn_model.h5")


def test():
    '''
    Test the model, need model.h5 -> run the train module 
    before testing 
    '''
    srcnn_model = model()
    srcnn_model.load_weights("./model/srcnn_model.h5")

    img = cv2.imread(IMG_NAME)
    shape = img.shape
    img = cv2.resize(
        img, (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC)
    img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    cv2.imwrite(BICUBIC_NAME, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0]
    pre = srcnn_model.predict(Y, batch_size=1)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    cv2.imwrite(OUTPUT_NAME, img)

    # PSNR and MSE calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread(BICUBIC_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

    print("test completed... below is the report")
    print("PSNR bicubic: ", psnr.psnr(im2, im1))
    print("PSNR SRCNN: ", psnr.psnr(im3, im1))
    print("MSE  bicubic: ", psnr.mse(im2, im1))
    print("MSE  srcnn: ", psnr.mse(im3, im1))


def apply_sr():
    '''
    Apply the model to image and result is the output
    '''
    srcnn_model = model()
    srcnn_model.load_weights("./model/srcnn_model.h5")

    img = cv2.imread(IMG_NAME)
    shape = img.shape
    #img = cv2.resize(
    #    img, (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)
    #img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img = cv2.resize(
        img, (scale * shape[1], scale * shape[0]), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0]
    pre = srcnn_model.predict(Y, batch_size=1)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    cv2.imwrite(OUTPUT_NAME, img)
    print("upscale by factor: ", scale, " done. Ouput Image: ", OUTPUT_NAME)


def display():
    '''
    For displaying the image loaded as input
    '''
    img = cv2.imread(IMG_NAME)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def set_variables(argv):
    '''
    Set all the command line options here
    '''
    from distutils.util import strtobool
    global IMG_NAME, BICUBIC_NAME, OUTPUT_NAME, scale, denoise, is_train, is_test, is_display
    IMG_NAME = ''
    scale = 2
    denoise = False
    is_train = False
    is_test = False
    is_display = False
    try:
        opts, args = getopt.getopt(
            argv, "hi:s:d:", ["help", "input=", "scale=", "denoise=", "train=", "test=", "display="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            IMG_NAME = arg
        elif opt in ("-s", "--scale"):
            scale = int(arg)
        elif opt in ("-d", "--denoise"):
            denoise = strtobool(arg)
        elif opt == "--train":
            is_train = strtobool(arg)
        elif opt == "--test":
            is_test = strtobool(arg)
        elif opt == "--display":
            is_display = strtobool(arg)
    if not(is_train) and IMG_NAME == '':
        print_usage()
        sys.exit(2)
    BICUBIC_NAME = IMG_NAME[0:IMG_NAME.rfind('.')] + "_bicubic.bmp"
    OUTPUT_NAME = IMG_NAME[0:IMG_NAME.rfind('.')] + "_output.bmp"


def print_usage():
    '''
    Print usage module
    '''
    print("Usage: main.py [options]\nwhere options include:")
    print("  -h  --help      prints this help message.")
    print("  -i  --input     input image name (*Mandatory for test and apply_sr)")
    print("  -s  --scale     scaling factor for image, should be integer. Default 2.")
    print(
        "  -d  --denoise   flag to enable denoise [True/False]. Default False.")
    print("      --train     Trains the model [True/False]. Default False.")
    print("      --test      Tests the model [True/False]. Default False.")
    print(
        "      --display   Display the input image provided [True/False]. Default False.")


if __name__ == "__main__":
    '''
    entry point of the program
    '''
    set_variables(sys.argv[1:])
    if is_train:
        print("training the model...")
        train()
        print("training completed...")
    elif is_test:
        print("testing the model...")
        test()
        print("testing completed...")
    elif is_display:
        print("displaying input image : ", IMG_NAME)
        display()
    else:
        print("main application...\nImage: ", IMG_NAME)
        apply_sr()
        print("Program completed.")
