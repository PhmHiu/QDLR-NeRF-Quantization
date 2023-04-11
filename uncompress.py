from __future__ import print_function
import argparse
import numpy as np
import time


from torch.utils.data import DataLoader
from coder import Decoder
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compressed', required=True, help='path to compressed file')

    configargs = parser.parse_args()
    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    decoder = Decoder()
    start = time.time()
    print('Start decoding...')
    model = decoder.decode(configargs.compressed)
    print('Done!', time.time()-start, 'seconds')
    if device == "cuda":
        model = model.cuda()
    model.eval()
    decoder.draw_distribution(10000, 2000)



if __name__ == "__main__":
    main()
