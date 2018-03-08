#!/usr/bin/env python
from utils.parsers import create_gen_parser
from models.gans.d48_resnet_dcgan import generator_forward
from utils.generate import generate_steps_png


if __name__ == "__main__":
    parser = create_gen_parser()
    parser.add_argument("--gen-dim", dest="gen_dim", default=64, type=int)
    parser.add_argument("save_dir", metavar="SAVE_DIR")

    generate_steps_png(parser.parse_args(), generator_forward)