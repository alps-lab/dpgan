from utils.parsers import create_gen_parser
from models.gans.d64_resnet_dcgan import generator_forward
from utils.generate import generate_steps


if __name__ == "__main__":
    parser = create_gen_parser()
    parser.add_argument("--gen-dim", dest="gen_dim", default=64, type=int)

    generate_steps(parser.parse_args(), generator_forward)