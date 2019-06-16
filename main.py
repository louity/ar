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


while continue_loop:
    input('Press Enter to start completion...')

    camera_config = config['camera']
    camera_url = camera_config['device_url']
    simulate_camera = camera_config['simulate']
    print('Taking picure with camera {} {}...'.format(camera_url, 'simulating' if simulate_camera else ''))
    img = utils_camera.take_picture(camera_url, simulate=simulate_camera)
    x_min = int(camera_config['x_min'])
    x_max = int(camera_config['x_max'])
    y_min = int(camera_config['y_min'])
    y_max = int(camera_config['y_max'])
    img = picture[x_min:x_max, y_min:y_max]

    print('Converting picture to strokes...')
    strokes_config = config['strokes']
    red = np.array(ast.literal_eval(strokes_config['red'])).reshape((1,1,3))
    blue = np.array(ast.literal_eval(strokes_config['blue'])).reshape((1,1,3))
    green = np.array(ast.literal_eval(strokes_config['green'])).reshape((1,1,3))
    colors = [red, blue, green]

    red_threshold = strokes_config['red_threshold']
    blue_threshold = strokes_config['blue_threshold']
    green_threshold = strokes_config['green_threshold']
    thresholds = [red_threshold, blue_threshold, green_threshold]

    color_images = utils.separate_color_images(img_np, colors, thresholds)


    print('Selecting a stroke...')
    color_id = np.random.randint(len(colors))


    print('Completing the stroke...')

    print('Displaying the completed stroke...')



