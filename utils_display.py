import matplotlib.pyplot as plt
import numpy as np
import PIL

def save_stroke_image(stroke, screen_height, screen_width, image_y, image_x, image_height, image_width):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.plot(stroke[:, 1], stroke[:, 0])

    fig.savefig('tmp.jpg', bbox_inches='tight', pad_inches=0, dpi=500)
    img = np.array(PIL.Image.open('./tmp.jpg'))
    full_img = 255*np.ones((screen_height, screen_width, 3), dtype=np.uint8)
    full_img[image_y:image_y+image_height, image_x:image_x+image_width,:] = scipy.misc.imresize(img, (image_height, image_width))
    im = PIL.Image.fromarray(full_img)
    im.save('to_project.jpg')
