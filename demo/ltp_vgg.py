# Using LTPTextDetector & Vgg text recognizer to realize end2end
import sys

ltp_root = '/home/aaron/projects/LTPTextDetector'
sys.path.insert(0, ltp_root)
import words_detect as wd

import os
cwd = os.getcwd()
target_img = '../images/slogn_01.jpg'

# ----------------------- detect ----------------------------
img_path = cwd + '/' + target_img
print 'image path: ', img_path
print 'cwd: ', cwd
print 'chdir: ', ltp_root
os.chdir(ltp_root)
detect_result = wd.detect_words(img_path)
token1 = ' '
token2 = ';'
rect_list = []
if detect_result is not None:
    word_rects = detect_result.split(token2)
    for word_rect in word_rects:
        segs = word_rect.split(token1)
        rect = []
        for i in range(len(segs)):
            rect.append(int(segs[i]))
        rect_list.append(rect)
for rect in rect_list:
    print rect

# ----------------------- recognize --------------------------
import caffe
import numpy as np
print 'chdir: ', cwd
os.chdir(cwd)

deploy = '/data/vgg/dictnet_vgg_deploy.prototxt'
model = '/data/vgg/dictnet_vgg.caffemodel'
labels_file = '/data/vgg/dictnet_vgg_labels.txt'

caffe.set_device(0)
caffe.set_mode_gpu()

labels = np.loadtxt(labels_file, str, delimiter='\t')
net = caffe.Classifier(deploy, model, image_dims=(32, 100), raw_scale=255)
image = caffe.io.load_image(target_img, False)

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
