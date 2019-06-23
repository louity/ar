import ast
import argparse
import configparser
import matplotlib.pyplot as plt
import numpy as np
import utils
import utils_camera
import utils_RNN
import utils_display

from sketchRNN.sketch_rnn import HParams


parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='config_example.ini', help='path to the configuration file')
parser.add_argument('--debug', action='store_true', help='plot successive actions')
args = parser.parse_args()


config = configparser.ConfigParser()
config.read(args.config_file)

continue_loop = True

while continue_loop:
    input('Press Enter to start completion...')

    camera_config = config['camera']
    camera_path = camera_config['path'].replace('PERCENT', '%')
    camera_path = '.'

    print('Selecting the last picture in folder {}'.format(camera_path))
    img = utils_camera.get_latest_img(camera_path)

    if args.debug:
        plt.figure()
        plt.imshow(img)
        plt.show()

    x_min = int(camera_config['x_min'])
    x_max = int(camera_config['x_max'])
    y_min = int(camera_config['y_min'])
    y_max = int(camera_config['y_max'])

    img = img[y_min:y_max, x_min:x_max]
    if args.debug:
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
    subsample = int(strokes_config['subsample'])
    min_length = int(strokes_config['min_length'])

    color_id = np.random.randint(len(colors))
    color_img = color_images[color_id]
    color_strokes = utils.get_strokes(color_img, subsample=subsample, min_length=min_length)
    stroke_id = np.random.randint(len(color_strokes))
    stroke = color_strokes[stroke_id]


    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(color_img)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    aspect = ax1.get_aspect()
    if args.debug:
        ax2.set_xlim(xlim);
        ax2.set_ylim(ylim);
        ax2.set_aspect(aspect);
        stroke = np.array(stroke)
        ax2.plot(stroke[:,1], stroke[:,0])
        plt.show()


    print('Completing the stroke...')
    stroke = np.array(stroke)
    sketchRNN_config = config['sketchRNN']
    new_strokes = utils_RNN.complete_stroke(stroke, sketchRNN_config)

    if args.debug:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(color_img)
        ax2.set_xlim(xlim);
        ax2.set_ylim(ylim);
        ax2.set_aspect(aspect);
        for new_stroke in new_strokes:
            ax2.plot(new_stroke[:,1], new_stroke[:,0])
        plt.show()

    spline_order = int(sketchRNN_config['spline_order'])
    n_points_interpolation = int(sketchRNN_config['n_points_interpolation'])

    interpolated_new_strokes = [
        utils.interpolate_stroke(new_stroke, spline_order=spline_order, n_points=n_points_interpolation)
        for new_stroke in new_strokes]


    print('Displaying the completed stroke...')
    display_config = config['display']

    screen_width = int(display_config['screen_width'])
    screen_height = int(display_config['screen_height'])
    image_x = int(display_config['image_x'])
    image_y = int(display_config['image_y'])
    image_width = int(display_config['image_width'])
    image_height = int(display_config['image_height'])
    stroke_width = float(display_config['stroke_width'])

    utils_display.save_stroke_image(interpolated_new_strokes, xlim, ylim, aspect, screen_height, screen_width, image_y, image_x, image_height, image_width, stroke_width)
    utils_display.display_stroke()

    input_text = input('Write "stop" or press Enter to continue...')
    if input_text == 'stop':
        continue_loop = False

