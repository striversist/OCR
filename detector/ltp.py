# Using LTPTextDetector & Vgg text recognizer to realize end2end
import sys

ltp_root = '/home/aaron/projects/LTPTextDetector'
sys.path.append(ltp_root)
import words_detect as wd

import os
cwd = os.getcwd()

def detect(img_path):
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
    os.chdir(cwd)
    return rect_list