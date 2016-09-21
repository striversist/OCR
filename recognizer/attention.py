import sys
import os

cwd = os.getcwd()
attention_root = '/home/aaron/projects/Attention-OCR'
sys.path.append(attention_root)
sys.path.append(attention_root + '/src')
import predictor


def recognize(img_path, rect_list):
    os.chdir(attention_root)
    boxes = []
    for rect in rect_list:
        box = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        boxes.append(box)

    words = predictor.predict(img_path, boxes)
    os.chdir(cwd)
    return words
