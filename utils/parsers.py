import argparse


def create_gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_path", metavar="MODEL_PATH")
    parser.add_argument("-b", "--batch-size", dest="batch_size", default=64, type=int)
    parser.add_argument("-t", "--times", dest="times", default=7, type=int)
    parser.add_argument("-s", "--save-path", dest="save_path")

    return parser


def create_nodp_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", dest="batch_size", default=64, type=int)
    parser.add_argument("-l", "--lambda", dest="lambd", default=10.0, type=float)
    parser.add_argument("-e", "--num-epoch", dest="num_epoch", default=10, type=int)
    parser.add_argument("-c", "--num-critic", dest="critic_iters", default=5, type=int)

    parser.add_argument("--load-path", dest="load_path")
    parser.add_argument("--image-dir", dest="image_dir")
    parser.add_argument("--save-dir", dest="save_dir")
    parser.add_argument("--image-every", dest="image_every", type=int, default=20)
    parser.add_argument("--save-every", dest="save_every", type=int, default=200)
    parser.add_argument("--total-step", dest="total_step", type=int)

    parser.add_argument("-g", "--num-gpu", dest="num_gpu", type=int, default=1)
    return parser


def create_dp_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", dest="batch_size", default=64, type=int)
    parser.add_argument("-l", "--lambda", dest="lambd", default=10.0, type=float)
    parser.add_argument("-e", "--num-epoch", dest="num_epoch", default=10, type=int)
    parser.add_argument("-c", "--num-critic", dest="critic_iters", default=5, type=int)

    parser.add_argument("--load-path", dest="load_path")
    parser.add_argument("--image-dir", dest="image_dir")
    parser.add_argument("--save-dir", dest="save_dir")
    parser.add_argument("--image-every", dest="image_every", type=int, default=20)
    parser.add_argument("--save-every", dest="save_every", type=int, default=1000)
    parser.add_argument("--gan-load-path", dest="gan_load_path")

    parser.add_argument("-g", "--num-gpu", dest="num_gpu", type=int, default=1)

    parser.add_argument("-a", "--accounting", action="store_true", dest="enable_accounting")
    parser.add_argument("-s", "--sigma", dest="sigma", type=float, default=1.0 / 4.0)
    parser.add_argument("-n", "--max-norm", dest="C", type=float, default=4.0)
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=1.0)
    parser.add_argument("--delta", dest="delta", type=float, default=1e-4)
    parser.add_argument("--target-deltas", dest="target_deltas", type=float, default=1e-5, nargs="*")
    parser.add_argument("--target-epsilons", dest="target_epsilons", type=float, default=1.0, nargs="*")
    parser.add_argument("--keep-sigma", dest="keep_sigma", action="store_true")
    parser.add_argument("--log-path", dest="log_path")
    parser.add_argument("--log-every", dest="log_every", type=int, default=10)
    parser.add_argument("--moment", default=24, type=int, dest="moment")
    parser.add_argument("--terminate", action="store_true", dest="terminate")

    parser.add_argument("--clipper", dest="clipper", default="basic")
    parser.add_argument("--scheduler", dest="scheduler", default="basic")
    parser.add_argument("--total-step", dest="total_step", type=int)

    return parser
