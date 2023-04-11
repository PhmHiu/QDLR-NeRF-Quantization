import argparse
import time

import torch
import json
from coder import Encoder
from nerf import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file."
    )
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        help="Path to trained network",
    )
    parser.add_argument(
        "--out", type=str, help="Path to compressed network file"
    )
    configargs = parser.parse_args()

    # Read config file.
    config = json.load(open(configargs.config,'r'))

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    

    # Initialize model.
    model = getattr(models, config['type'])(
        num_layers=config['num_layers'],
        hidden_size=config['hidden_size'],
        skip_connect_every=config['skip_connect_every'],
        num_encoding_fn_xyz=config['num_encoding_fn_xyz'],
        num_encoding_fn_dir=config['num_encoding_fn_dir'],
        include_input_xyz=config['include_input_xyz'],
        include_input_dir=config['include_input_dir'],
        use_viewdirs=config['use_viewdirs'],
    )
    model.to(device)
    model.load_state_dict(torch.load(configargs.network, map_location=torch.device(device)))
    model.eval()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    encoder = Encoder(model, config)
    start = time.time()
    print('Start encoding...')
    encoder.encode(configargs.out, 256)
    print('Done!', time.time()-start, 'seconds')
    encoder.draw_distribution(10000, 2000)

if __name__ == "__main__":
    main()
