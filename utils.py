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


def denoise_tv(img, tv_weight=1, max_ratio=0.3):
    img_tv = skimage.restoration.denoise_tv_chambolle(img, weight=tv_weight)
    img_tv = img_tv > (max_ratio * img_tv.max())
    return img_tv


def get_sketch(img, min_length=5, subsample=5, max_dist_in_stroke=3):
    """Generates the sketch of a binary image."""
    skeleton = skimage.morphology.skeletonize(img)
    labeled_skeleton = skimage.morphology.label(skeleton, background=0, connectivity=2)
    sketch = []
    for i_label in range(1, labeled_skeleton.max()+1):
        strokes_mask = labeled_skeleton == i_label
        strokes = convert_mask_to_strokes(strokes_mask, img, max_dist=max_dist_in_stroke)
        strokes = [stroke[::subsample] for stroke in strokes]
        strokes = [stroke for stroke in strokes if len(stroke) > min_length]
        if len(strokes) > 0:
            sketch.append(strokes)

    return sketch

def interpolate_stroke(stroke, spline_order, n_points):
    """
    Parameters
        stroke: np array, shape (n, 2)

        spline_order: int
            integer order of the spline

        n_points: int
            number of points for the interpolation

    Returns
        interpolated_stroke: np array shape (n_points, 2)
    """

    tck, u = scipy.interpolate.splprep(stroke.transpose(), s=spline_order)
    unew = np.linspace(0, 1, n_points)
    out = scipy.interpolate.splev(unew, tck)

    interpolated_stroke = np.array(out).transpose()

    return interpolated_stroke


def translate_and_rotate_stroke(stroke, img_shape):
    theta = np.random.uniform(0, 2*np.pi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    R = np.array(((cos_theta,-sin_theta), (sin_theta, cos_theta)))

    rotated_stroke = stroke.copy().astype('float32')
    mean = rotated_stroke.mean(axis=0, keepdims=True)
    rotated_stroke -= mean
    rotated_stroke = np.dot(rotated_stroke, R.transpose())
    rotated_stroke += mean

    tx_min, tx_max = -rotated_stroke[:, 1].min(), img_shape[1] - rotated_stroke[:, 1].max()
    ty_min, ty_max = -rotated_stroke[:, 0].min(), img_shape[0] - rotated_stroke[:, 0].max()
    tx, ty = np.random.uniform(tx_min, tx_max), np.random.uniform(ty_min, ty_max)
    translated_stroke = rotated_stroke.copy()
    translated_stroke[:,1] += tx
    translated_stroke[:,0] += ty

    return translated_stroke.astype('int')


def periodize_stroke(stroke, xlim, ylim):
    # FIXME: not working yet
    periodized_stroke = np.zeros_like(stroke)
    periodized_stroke[:,0] = ylim[0] + (stroke[:, 0] - ylim[0]) % (ylim[1] - ylim[0])
    periodized_stroke[:,1] = xlim[0] + (stroke[:, 1] - xlim[0]) % (xlim[1] - xlim[0])
    return periodized_stroke

color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

