
import math
import sys
import getopt
import cv2
import numpy


def psnr(target, ref):
    '''
    assume RGB image, we calculate the PSNR between two images
    NOTE: the two images must have the same dimension
    '''
    rmse = math.sqrt(mse(target, ref))
    return 20 * math.log10(255. / rmse)


def mse(target, ref):
    '''
    the 'Mean Squared Error' between the two images is the
    sum of the squared difference between the two images;
    NOTE: the two images must have the same dimension
    '''
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    return numpy.mean(diff ** 2.)


def set_variables(argv):
    '''
    Set all the command line options here
    '''
    global IMG1_NAME, IMG2_NAME
    IMG1_NAME = ''
    IMG2_NAME = ''
    try:
        opts, args = getopt.getopt(
            argv, "hi:o:", ["help", "image1=", "image2="])
    except getopt.GetoptError:
        print("usage: psnr.py -i <image1> -o <image2>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("usage: psnr.py -i <image1> -o <image2>")
            sys.exit()
        elif opt in ("-i", "--image1"):
            IMG1_NAME = arg
        elif opt in ("-o", "--image2"):
            IMG2_NAME = arg

    if IMG1_NAME == '' or IMG2_NAME == '':
        print("usage: psnr.py -i <image1> -o <image2>")
        sys.exit(2)


if __name__ == "__main__":
    '''
    entry point of the program
    '''
    set_variables(sys.argv[1:])
    print("Image1 file is ", IMG1_NAME)
    img1 = cv2.imread(IMG1_NAME, cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    print("Image2 file is ", IMG2_NAME)
    img2 = cv2.imread(IMG2_NAME, cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

    print("PSNR SRCNN: ", psnr(img2, img1))
    print("MSE  srcnn: ", mse(img2, img1))
