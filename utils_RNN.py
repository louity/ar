from sketchRNN.complete_img import tina_et_charlie_2
import torch

use_cuda = torch.cuda.is_available()

def complete_stroke(stroke, sketchRNN_config):
    hp_filepath = sketchRNN_config['hp_filepath']
    encoder_ckpt = sketchRNN_config['encoder_ckpt']
    decoder_ckpt = sketchRNN_config['decoder_ckpt']
    nbr_point_next = int(sketchRNN_config['nbr_point_next'])
    sigma = float(sketchRNN_config['sigma'])
    set_first_point_to_zero = sketchRNN_config['set_first_point_to_zero'] == 'True'
    rescale_tail = sketchRNN_config['rescale_tail'] == 'True'

    paint = [stroke]
    new_strokes = tina_et_charlie_2(hp_filepath, encoder_ckpt, decoder_ckpt, use_cuda,
                                    nbr_point_next, paint, paint, sigma,
                                    set_first_point_to_zero=set_first_point_to_zero,
                                    rescale_tail=rescale_tail)
    return new_strokes
