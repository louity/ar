import ast
import argparse
import configparser
import numpy as np
import PIL
import scipy
import skimage
import torch
import utils
import utils_camera
import utils_RNN
import utils_display

from sketchRNN.sketch_rnn import HParams


parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='config_example.ini', help='path to the configuration file')
parser.add_argument('--model-name', default='broccoli_car_cat_20000', help='name of the model')
parser.add_argument('--nbr-point-next', type=int, default=10, help='number of points to complete the stroke')
parser.add_argument('--sigma', type=float, default=0.1, help='sigma parameter for the sampling')
args = parser.parse_args()


config = configparser.ConfigParser()
config.read(args.config_file)

continue_loop = True

while continue_loop:
    input('Press Enter to start completion...')

    camera_config = config['camera']
    camera_url = camera_config['device_url']
    simulate_camera = camera_config['simulate'] == 'True'
    print('Taking picure with camera {} {}...'.format(camera_url, 'simulating' if simulate_camera else ''))
    img = utils_camera.take_picture(camera_url, simulate=simulate_camera)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(img)
    plt.show()

    x_min = int(camera_config['x_min'])
    x_max = int(camera_config['x_max'])
    y_min = int(camera_config['y_min'])
    y_max = int(camera_config['y_max'])

    img = img[y_min:y_max, x_min:x_max]
    plt.figure()
    plt.imshow(img)
    plt.show()

    print('Converting picture to strokes...')
    strokes_config = config['strokes']
    red = np.array(ast.literal_eval(strokes_config['red'])).reshape((1,1,3))
    blue = np.array(ast.literal_eval(strokes_config['blue'])).reshape((1,1,3))
    green = np.array(ast.literal_eval(strokes_config['green'])).reshape((1,1,3))
    colors = [red, blue, green]

    red_threshold = int(strokes_config['red_threshold'])
    blue_threshold = int(strokes_config['blue_threshold'])
    green_threshold = int(strokes_config['green_threshold'])
    thresholds = [red_threshold, blue_threshold, green_threshold]

    color_images = utils.separate_color_images(img, colors, thresholds)

    print('Selecting a stroke...')
    color_id = np.random.randint(len(colors))
    color_img = color_images[color_id]
    color_strokes = utils.get_strokes(color_img)
    stroke_id = np.random.randint(len(color_strokes))
    stroke = color_strokes[stroke_id]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(color_img)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    aspect = ax1.get_aspect()
    ax2.set_xlim(xlim);
    ax2.set_ylim(ylim);
    ax2.set_aspect(aspect);
    stroke = np.array(stroke)
    ax2.plot(stroke[:,1], stroke[:,0])
    plt.show()


    print('Completing the stroke...')
    paint = [stroke]
    new_strokes = utils_RNN.complete_stroke(stroke, args)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(color_img)
    ax2.set_xlim(xlim);
    ax2.set_ylim(ylim);
    ax2.set_aspect(aspect);
    for new_stroke in new_strokes:
        ax2.plot(new_stroke[:,1], new_stroke[:,0])
    plt.show()

    import pdb;pdb.set_trace()

    print('Displaying the completed stroke...')
    # TODO

    input_text = input('Write "stop" or press Enter to continue...')
    if input_text == 'stop':
        continue_loop = False

