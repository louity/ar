from sketchRNN.complete_img import tina_et_charlie_2
import torch

use_cuda = torch.cuda.is_available()

def complete_stroke(stroke, args):
    hp_filepath = 'sketchRNN/draw_models/hp_folder/{}.pickle'.format(args.model_name)
    encoder_ckpt = 'sketchRNN/draw_models/encoder_{}.pth'.format(args.model_name)
    decoder_ckpt = 'sketchRNN/draw_models/decoder_{}.pth'.format(args.model_name)

    paint = [stroke]

    new_strokes = tina_et_charlie_2(hp_filepath, encoder_ckpt, decoder_ckpt, use_cuda,
                                      args.nbr_point_next, paint, paint, args.sigma)
    return new_strokes
