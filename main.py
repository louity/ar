import ast
import argparse
import configparser
import numpy as np
import utils
import utils_camera

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='config_example.ini', help='path to the configuration file')
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
    ax2.set_xlim(ax1.get_xlim());
    ax2.set_ylim(ax1.get_ylim());
    ax2.set_aspect(ax1.get_aspect());
    stroke = np.array(stroke)
    ax2.plot(stroke[:,1], stroke[:,0])
    plt.show()

    print('Completing the stroke...')

    print('Displaying the completed stroke...')

    input_text = input('Write "stop" or press Enter to continue...')
    if input_text == 'stop':
        continue_loop = False


