import argparse
import ast


def make(*args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="Facenet", type=str, help="")
    parser.add_argument("--random_seed", default=0, type=int, help="")

    #general testing arguments
    parser.add_argument("--backdoor_type", default="sc", type=str, help="")
    parser.add_argument("--reinflate", action="store_true", help="")
    parser.add_argument("--normalise", action="store_true", help="")
    parser.add_argument("--pca_normalise", action="store_true", help="")

    #mode
    parser.add_argument("--extract_celeba", action="store_true", help="")
    parser.add_argument("--model_summary", action="store_true", help="")
    parser.add_argument("--pca_test", action="store_true", help="")
    parser.add_argument("--get_eigenvalues", action="store_true", help="")
    parser.add_argument("--eps_delta", action="store_true", help="")
    parser.add_argument("--angle_dist", action="store_true", help="")
    parser.add_argument("--get_normalised_eigenvalues", action="store_true", help="")

    return parser.parse_args(args)
