# Use LTPTextDetector as detector, AttentionOCR as recognizer to realize end2end
import sys
sys.path.append('../')
from detector import ltp
from recognizer import attention
import argparse


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', dest="image_path",
                        type=str, help=('Image file path'))
    parameters = parser.parse_args(args)

    img_path = parameters.image_path
    print 'input image path: ', img_path
    rect_list = ltp.detect(img_path)
    words = attention.recognize(img_path, rect_list)
    for word in words:
        print word


if __name__ == "__main__":
    main(sys.argv[1:])