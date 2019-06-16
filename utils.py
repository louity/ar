import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import skimage.morphology

def get_axis_bounds(axis):
    x_min, x_max = axis.get_xlim()
    y_max, y_min = axis.get_ylim()
    x_min, y_min = np.floor(x_min).astype('int'), np.floor(y_min).astype('int')
    x_max, y_max = np.ceil(x_max).astype('int'), np.ceil(y_max).astype('int')
    return x_min, x_max, y_min, y_max

def plot_color_distance_histograms(img, colors,  bins=100):
    fig, axes = plt.subplots(len(colors))
    fig.suptitle('Histograms of distance w.r.t colors')
    for i_color, color in enumerate(colors):
        axes[i_color].hist(np.abs(img - color).sum(axis=2).ravel(), bins=bins);


def separate_color_images(img, colors, thresholds):
    color_images = []
    for color, threshold in zip(colors, thresholds):
         color_images.append((np.abs(img - color).sum(axis=2) < threshold).astype('float'))
    return color_images


def get_connected_components(img, i=None):
    connected_components = []
    labeled_components = skimage.measure.label(img, background=0)
    if i is not None:
        return (labeled_components == i).astype('float')

    for label in range(1, len(labeled_components)):
        connected_components.append((labeled_components == label).astype('float'))
    return connected_components


def zoom_on_nonzero(img, padding=10):
    nonzero = np.argwhere(img > 0)
    x_min = max(nonzero[:,0].min() - padding, 0)
    x_max = min(nonzero[:,0].max() + padding, img.shape[0])
    y_min = max(nonzero[:,1].min() - padding, 0)
    y_max = min(nonzero[:,1].max() + padding, img.shape[1])

    return img[x_min:x_max, y_min:y_max]


def is_connected(point_1, point_2, component):
    rr, cc = skimage.draw.line(point_1[0], point_1[1], point_2[0], point_2[1])
    return np.all(component[rr, cc])


def square_dist(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2


def convert_mask_to_strokes(mask, component, max_dist=None):
    point_list = np.argwhere(mask > 0).tolist()
    strokes = [[]]
    while len(point_list) > 0:
        stroke = strokes[-1]
        if len(stroke) == 0:
            stroke.append(point_list.pop(np.random.randint(len(point_list))))
            continue

        p_0 = stroke[0]
        p_last = stroke[-1]

        connected_points_0 = [(i, point) for (i, point) in enumerate(point_list) if is_connected(p_0, point, component)]
        connected_points_last = [(i, point) for (i, point) in enumerate(point_list) if is_connected(p_last, point, component)]

        if max_dist is not None:
            connected_points_0 = [(i, point) for (i, point) in connected_points_0 if square_dist(p_0, point) < max_dist**2]
            connected_points_last = [(i, point) for (i, point) in connected_points_last if square_dist(p_last, point) < max_dist**2]
        if len(connected_points_0) == 0 and len(connected_points_last) == 0:
            if len(point_list) > 0:
                strokes.append([])
        elif len(connected_points_0) == 0:
            closest_point = min(connected_points_last, key=lambda x, p=p_last: (x[1][0]-p[0])**2 + (x[1][1]-p[1])**2)
            stroke.append(point_list.pop(closest_point[0]))
        elif len(connected_points_last) == 0:
            closest_point = min(connected_points_0, key=lambda x, p=p_0: (x[1][0]-p[0])**2 + (x[1][1]-p[1])**2)
            stroke.insert(0, point_list.pop(closest_point[0]))
        else:
            closest_point_last = min(connected_points_last, key=lambda x, p=p_last: (x[1][0]-p[0])**2 + (x[1][1]-p[1])**2)
            dist_last = (closest_point_last[1][0] - p_last[0])**2 + (closest_point_last[1][1] - p_last[1])**2
            closest_point_0 = min(connected_points_0, key=lambda x, p=p_0: (x[1][0]-p[0])**2 + (x[1][1]-p[1])**2)
            dist_0 = (closest_point_0[1][0] - p_0[0])**2 + (closest_point_0[1][1] - p_0[1])**2
            if dist_0 < dist_last:
                stroke.insert(0, point_list.pop(closest_point_0[0]))
            else:
                stroke.append(point_list.pop(closest_point_last[0]))
    return strokes

def compute_maxima_mask(img):
    return ((img > 0) * (img == scipy.ndimage.filters.maximum_filter(img, size=3)))


neighbor_filter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype('float')

def compute_isolated_mask(img):
    return (img > 0) * (scipy.signal.convolve2d(img, neighbor_filter, mode='same') == 0)


def find_middle_lines_with_gaussian_blur(component, sigma_blur):
    component_blur = scipy.ndimage.gaussian_filter(component, sigma=sigma_blur)
    maxima_mask = compute_maxima_mask(component_blur)
    strokes = convert_mask_to_strokes(maxima_mask, component)
    return strokes


def find_middle_lines_with_successive_gaussian_blurs(component, sigmas):
    maxima_mask = np.zeros_like(component, dtype='bool')
    for sigma_blur in sigmas:
        component_blur = scipy.ndimage.gaussian_filter(component, sigma=sigma_blur)
        maxima_mask_sigma = compute_maxima_mask(component_blur).astype('float')
        isolated_maxima_mask = compute_isolated_mask(maxima_mask_sigma)
        maxima_mask += isolated_maxima_mask.astype('bool')

    strokes = convert_mask_to_strokes(maxima_mask, component)
    return strokes


def get_strokes(img, min_length=10):
    strokes = []
    labeled_components = skimage.measure.label(img, background=0)

    for label in range(1, labeled_components.max()):
        connected_component = (labeled_components == label).astype('float')
        skeleton = skimage.morphology.skeletonize(connected_component)
        strokes += convert_mask_to_strokes(skeleton, connected_component, max_dist=3)

    strokes = [stroke for stroke in strokes if len(stroke) > min_length]

    return strokes
