# Using LTPTextDetector & Vgg text recognizer to realize end2end
import sys

sys.path.append('../')
from detector import ltp
from recognizer import vgg
import argparse


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', dest="image_path",
                        type=str, help=('Image file path'))
    parameters = parser.parse_args(args)

    img_path = parameters.image_path
    print 'input image path: ', img_path
    vgg.recognize(img_path, ltp.detect(img_path))


if __name__ == "__main__":
    main(sys.argv[1:])
