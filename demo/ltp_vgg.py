# Using LTPTextDetector & Vgg text recognizer to realize end2end
import sys
import os

sys.path.append('../')
from detector import ltp

cwd = os.getcwd()

# ----------------------- recognize --------------------------
import caffe
import numpy as np


def recognize(img_path, rect_list):
    deploy = '/data/vgg/dictnet_vgg_deploy.prototxt'
    model = '/data/vgg/dictnet_vgg.caffemodel'
    labels_file = '/data/vgg/dictnet_vgg_labels.txt'

    caffe.set_device(0)
    caffe.set_mode_gpu()

    labels = np.loadtxt(labels_file, str, delimiter='\t')
    net = caffe.Classifier(deploy, model, image_dims=(32, 100), raw_scale=255)
    image = caffe.io.load_image(img_path, False)

    for rect in rect_list:
        # coordinates of rect (x1,y1, x2,y2) for word 'food' in picture img_20.jpg
        word_rect = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        word_part = image[word_rect[1]:word_rect[3], word_rect[0]:word_rect[2], :]
        prediction = net.predict([word_part], False)
        print '--------------------'
        print 'word rect:{}'.format(word_rect)
        print 'prediction shape: ', prediction[0].shape
        print 'predicted class:  ', prediction[0].argmax()
        print 'predicted label: ', labels[prediction[0].argmax()]


import argparse


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', dest="image_path",
                        type=str, help=('Image file path'))
    parameters = parser.parse_args(args)

    img_path = parameters.image_path
    print 'input image path: ', img_path
    recognize(img_path, ltp.detect(img_path))


if __name__ == "__main__":
    main(sys.argv[1:])
