import argparse
import pickle
import re

import numpy as np
import torch

from sketch_rnn import HParams
from sketch_rnn import Model
from sketch_rnn import DataLoader
from sketchrnn.visutils import make_image
from sketchrnn.batchutils import make_image_point


use_cuda = torch.cuda.is_available()


def make_seq(seq_x, seq_y, seq_z):
    '''
    To go from offset to plain coordinate
    '''
    x_sample = np.cumsum(seq_x, 0)
    y_sample = np.cumsum(seq_y, 0)
    z_sample = np.array(seq_z)
    sequence_coo = np.stack([x_sample, y_sample, z_sample]).T
    sequence_offset = np.stack([np.array(seq_x),
                               np.array(seq_y), np.array(z_sample)]).T
    return(sequence_coo, sequence_offset)


def from_larray_to_3array(l_array, continue_last_stroke=True):
    '''
    l_array : list of (nbr,2)
    '''
    x = []
    y = []
    z = []
    n_strokes = len(l_array)
    for id_stroke, stroke in enumerate(l_array):
        x.append(stroke[:, 0])
        y.append(stroke[:, 1])
        tab = np.zeros(len(stroke[:,0]))
        tab[-1] = 1
        if id_stroke == n_strokes-1 and continue_last_stroke:
            tab[-1] = 0
        z.append(tab)
    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)

    return np.stack([x,y,z], axis=1)


def from_3array_to_larray(array_3):
    l_idx_jump = list(np.where(array_3[:, 2] == 1)[0])

    strokes = []
    idx_old = 0
    for idx in l_idx_jump:
        stroke = array_3[idx_old:idx+1, :2].copy()
        idx_old = idx+1
        strokes.append(stroke)

    return strokes



def compute_variance(array_offsets):
    '''
    Method : Given a sequence of offset, compute the variance distinguishing
    between the jumps.
    Input : array_offsets is a (nbr,3) numpy representing (dx, dy, p)
    '''
    l_idx_jump = list(np.where(array_offsets[:, 2] == 1)[0])
    if l_idx_jump == []:
        norms = np.linalg.norm(array_offsets[:, :2], axis=1)
        return(norms.mean(), norms.std())
    else:
        mean = 0
        std = 0
        idx_old = 0
        for idx in l_idx_jump:
            norms = np.linalg.norm(array_offsets[idx_old: idx], axis=1)
            mean += norms.mean()
            std += norms.std()
            idx_old = idx

    mean = mean/len(l_idx_jump)
    std = std/len(l_idx_jump)

    return (mean, std)


def scale_stroke(array_offsets, scale):
    '''
    It put the std of each stroke to 1
    '''
    l_idx_jump = list(np.where(array_offsets[:, 2] == 1)[0])
    if l_idx_jump == []:
        return(array_offsets/scale)
    else:
        (nbr, _) = array_offsets.shape
        array_offsets_nor = np.zeros((nbr, 3))
        idx_old = 0
        for idx in l_idx_jump:
            array_offsets_nor[idx_old:idx, 0:2] = array_offsets[idx_old:idx, 0:2]/scale
            idx_old = idx
        array_offsets_nor[:, 2] = array_offsets[:, 2]
    return(array_offsets_nor)


def adjust_img(array_offsets):
        '''
        array_offsets : numpy (nbr,3) of format (x,y,p)
        '''
        img_full = array_offsets
        # from (x,y,p) to (dx,dy,p)
        img_full[1:, 0:2] = img_full[1:, 0:2] - img_full[:-1, 0:2]
        # get standard dev
        mean_full, std_full = compute_variance(img_full)
        # scale on each stroke
        img_full = scale_stroke(img_full, std_full)
        # from (dx,dy,p) to input of RNN
        img_full = make_image_point(img_full)
        return(img_full)



def generate_example_sketch(n_points=30):
    """Generates sketch with one half-circle stroke."""
    angle = np.linspace(0, np.pi, n_points)
    half_circle_stroke = np.zeros((n_points, 2))
    half_circle_stroke[:, 0] = np.cos(angle)
    half_circle_stroke[:, 1] = np.sin(angle)

    sketch = [half_circle_stroke]
    return sketch


def complete_sketch(hp_filepath, encoder_checkpoint, decoder_checkpoint,
                    use_cuda, nbr_point_next, painting_completing,
                    painting_conditioning, sigma=0.1, plot=False,
                    set_first_point_to_zero=False, rescale_tail=True, seed=None):
    """Complete existing sketch using pretrained sketchRNN.

    Parameters
        hp_filepath: string

        encoder_checkpoint: string

        decoder_checkpoint: string

        use_cuda: boolean

        nbr_point_next: int

        painting_completing:

        painting_conditioning:

        sigma: float

    Returns

    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    painting_completing = from_larray_to_3array(painting_completing)
    painting_conditioning = from_larray_to_3array(painting_conditioning)

    if set_first_point_to_zero:
        painting_completing_first_point = painting_completing[0:1].copy()
        painting_conditioning_first_point = painting_conditioning[0:1].copy()

        painting_completing -= painting_completing_first_point
        painting_conditioning -= painting_conditioning_first_point

    with open(hp_filepath, 'rb') as handle:
        hp = pickle.load(handle)
    hp.use_cuda = use_cuda

    # load model
    model = Model(hyper_parameters=hp, parametrization='point')
    model.load(encoder_checkpoint, decoder_checkpoint)

    # from format (x, y, p) to (dx, dy, p)
    datum = painting_completing
    datum[1:, 0:2] = datum[1:, 0:2] - datum[:-1, 0:2]

    # normalize
    mean_ini, std_ini = compute_variance(datum)
    datum_scaled = scale_stroke(datum, std_ini)
    # format from (dx,dy,p) to the 5
    img_to_complete = make_image_point(datum_scaled)

    # determining the image that will condition the latent vector z.
        # format (x,y,p) to (dx,dy,p)
    img_full = painting_conditioning
    img_full[1:, 0:2] = img_full[1:, 0:2] - img_full[:-1, 0:2]
    mean_full, std_full = compute_variance(img_full)
    img_full = scale_stroke(img_full, std_full)
    img_full = make_image_point(img_full)

    # complete
    img_tail = model.finish_drawing_point(img_to_complete,
                                          use_cuda,
                                          nbr_point_next=nbr_point_next,
                                          img_full=img_full,
                                          sigma=sigma)

    # process the tail so that it has the same variance as the images
    # it tries to complete.
    if rescale_tail:
        mean_tail, std_tail = compute_variance(img_tail)
        img_tail = scale_stroke(img_tail, std_tail)

    img_total = np.concatenate((datum_scaled, img_tail), 0)

    # rescale the images to the original scale
    img_tail *= std_ini
    img_total *= std_ini

    # plot the image..
    (img_coo, img_offset) = make_seq(img_total[:, 0],
                                     img_total[:, 1],
                                     img_total[:, 2])
    (img_tail_coo, img_tail_offset) = make_seq(img_tail[:, 0],
                                               img_tail[:, 1],
                                               img_tail[:, 2])


    if plot:
        make_image(img_coo, 1, dest_folder=None, name='_output_', plot=True)
        make_image(img_tail_coo, 2, dest_folder=None, name='_output_', plot=True)

    # set the end of the stroke
    img_coo[-1, 2] = 1
    img_tail_coo[-1, 2] = 1

    if set_first_point_to_zero:
        img_coo += painting_completing_first_point
        img_tail_coo += painting_completing_first_point

    new_strokes = from_3array_to_larray(img_coo)
    return new_strokes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drawing arguments')
    parser.add_argument('--hp_filepath', default='draw_models/hp_folder/broccoli_car_cat_20000.pickle')
    parser.add_argument('--encoder_checkpoint', default='draw_models/encoder_broccoli_car_cat_20000.pth')
    parser.add_argument('--decoder_checkpoint', default='draw_models/decoder_broccoli_car_cat_20000.pth')
    parser.add_argument('--sigma', default=1, type=float, help='variance of the gaussian')
    parser.add_argument('--nbr_point_next', default=30, type=int, help='nbr of point continuing the draw')
    parser.add_argument('--plot', action='store_true', help='plot result')
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    args = parser.parse_args()

    circle_sketch = generate_example_sketch()
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    img_completed = complete_sketch(args.hp_filepath, args.encoder_checkpoint, args.decoder_checkpoint, use_cuda,
                      args.nbr_point_next, circle_sketch, circle_sketch,
                      args.sigma, plot=args.plot, seed=0)
