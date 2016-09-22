from PIL import Image, ImageDraw, ImageFont


def show(image_path=None, boxes=[], words=[]):
    if image_path is None:
        print 'error: image path is invalid'
        return

    im = Image.open(image_path)
    draw = ImageDraw.ImageDraw(im)
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 20)
    for i, box in enumerate(boxes):
        coord_box = (box[0], box[1], box[0] + box[2], box[1] + box[3])
        draw.rectangle(coord_box, outline='red')
        draw.text((coord_box[0], coord_box[1]), text=words[i], font=font, fill='red')
    im.show()
