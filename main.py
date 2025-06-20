import utils
import csv
import matplotlib.pyplot as plt
import os
import random
import sys
from tqdm import tqdm
from lfw import read_pairs, get_paths
import pickle
import numpy as np
import torch
import parse_args


@torch.no_grad()
def accuracy_test(args):
    if args.reinflate:
        log_dir = f"runs/{args.model_name}/reinflate/{args.backdoor_type}/{args.random_seed}/"
    elif args.normalise:
        log_dir = f"runs/{args.model_name}/normalise/{args.backdoor_type}/{args.random_seed}/"
    elif args.pca_normalise:
        log_dir = f"runs/{args.model_name}/pca_normalise/{args.backdoor_type}/{args.random_seed}/"
    else:
        log_dir = f"runs/{args.model_name}/standard/{args.backdoor_type}/{args.random_seed}/"

    os.makedirs(log_dir, exist_ok=True)

    random.seed(args.random_seed)

    pairs = read_pairs(utils.LFW_PAIRS_PATH)
    lfw_paths, issame_list = get_paths(utils.LFW_DIR, pairs, "jpg")

    # Open lfw features
    lfw_dropout_features, _ = utils.extract_lfw_features(args)

    # Try with weight matrix
    W, W_bias = utils.get_weight_matrix(args.model_name)
    if args.normalise:
        W = utils.normalise_weight_matrix(W)

    bottleneck_features = utils.apply_weight_matrix(W, lfw_dropout_features, bias=W_bias)

    # Check if threshold is already saved
    if not os.path.exists(f"embeddings/{args.model_name}_thresholds.pkl"):
        thresholds = []
        for lfw_split in range(10):
            # Split into test and train
            train_issame_list, test_issame_list = utils.train_test_split(issame_list, lfw_split)
            train_bottleneck_features, _ = utils.train_test_split(bottleneck_features, lfw_split, double=True)

            # Find threshold
            threshold = utils.train_threshold(train_bottleneck_features, train_issame_list)
            thresholds.append(threshold)
        with open(f"embeddings/{args.model_name}_thresholds.pkl", "wb") as f:
            pickle.dump(thresholds, f)

    with open(f"embeddings/{args.model_name}_thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)

    # Check if best celeba_features are already saved
    if not os.path.exists(f"embeddings/{args.model_name}_best_celeba_features.pkl"):
        print("Finding best CelebA features")
        celeba_dropout_features, celeba_bottleneck_features = utils.extract_celeba_features(args)

        # Sort classes by number of images
        celeba_dropout_features = utils.find_best_celeba(celeba_dropout_features, celeba_bottleneck_features, threshold=thresholds[0])
        celeba_dropout_features = celeba_dropout_features[:1024]
        with open(f"embeddings/{args.model_name}_best_celeba_features.pkl", "wb") as f:
            pickle.dump(celeba_dropout_features, f)

    with open(f"embeddings/{args.model_name}_best_celeba_features.pkl", "rb") as f:
        celeba_dropout_features = pickle.load(f)

    random.shuffle(celeba_dropout_features)

    # Find CelebA bottleneck features
    celeba_bottleneck_features = []
    for d_f in celeba_dropout_features:
        celeba_bottleneck_features.append(utils.apply_weight_matrix(W, d_f))

    if args.pca_normalise:
        # Concatenate all bottleneck features
        concat_celeba = torch.cat(celeba_bottleneck_features, dim=0)
        print(f"Concatenated shape: {concat_celeba.shape}")
        W, W_bias = utils.pca_normalise_weight_matrix(args, W, concat_celeba, W_bias=W_bias)
        celeba_bottleneck_features = []
        for d_f in celeba_dropout_features:
            celeba_bottleneck_features.append(utils.apply_weight_matrix(W, d_f))

        bottleneck_features = utils.apply_weight_matrix(W, lfw_dropout_features, bias=W_bias)

    if args.backdoor_type == "pca":
        concat_celeba = torch.cat(celeba_bottleneck_features, dim=0)
        exp_var, eig_vecs = utils.pca(concat_celeba)

    # For each dimension
    num_classes_used = 0
    accuracies = []
    backdoor_logs = []
    for i in tqdm(range(utils.MODEL_SUMMARY[args.model_name]["Bottleneck_size"])):
        print(i)
    # for i in tqdm(range(10)):
        # Find projection matrix for a CelebA class
        if args.backdoor_type == "random":
            b_t = random.choice(["sc", "mc"])
        else:
            b_t = args.backdoor_type

        if b_t == "sc":
            P = utils.find_projection(celeba_bottleneck_features[num_classes_used])
            backdoor_logs.append(utils.add_backdoor_log(b_t, num_classes_used))
            num_classes_used += 1
        elif b_t == "mc":
            P = utils.merged_class_projection(celeba_bottleneck_features[num_classes_used], celeba_bottleneck_features[num_classes_used+1])
            backdoor_logs.append(utils.add_backdoor_log(b_t, num_classes_used))
            num_classes_used += 2
        elif b_t == "pca":
            P = utils.find_projection(eig_vecs[i], mean=False)

        # Calculate accuracies for all backdoor logs
        for b_l in backdoor_logs:
            index = b_l["indices"]
            if b_l["backdoor_type"] == "sc":
                b_l["accuracies"].append(utils.celeba_test_accuracy(celeba_bottleneck_features[index], threshold=thresholds[0]))
            elif b_l["backdoor_type"] == "mc":
                b_l["accuracies"].append(utils.celeba_test_accuracy_diff_classes(celeba_bottleneck_features[index], celeba_bottleneck_features[index+1], threshold=thresholds[0]))

        # Apply to W, make sure dimensions are correct
        W = torch.matmul(W, P.T)
        if args.reinflate:
            W = utils.reinflate_weight_matrix(W)

        # Find new bottleneck features
        celeba_bottleneck_features = []
        for d_f in celeba_dropout_features:
            celeba_bottleneck_features.append(utils.apply_weight_matrix(W, d_f))


        lfw_bottleneck_features = utils.apply_weight_matrix(W, lfw_dropout_features)

        # Test accuracy (of both CelebA class and LFW)
        curr_acc = []
        for lfw_split in range(10):
            _, test_issame_list = utils.train_test_split(issame_list, lfw_split)
            _, test_bottleneck_features = utils.train_test_split(lfw_bottleneck_features, lfw_split, double=True)

            acc = utils.calc_accuracy(test_bottleneck_features, test_issame_list, threshold=thresholds[lfw_split])
            curr_acc.append(acc)

        accuracies.append(curr_acc)

        # print(f"Accuracy: {accuracies}")
        # print(f"Backdoor logs: {backdoor_logs}")

    # Save accuracies and backdoor logs
    save_path = os.path.join(log_dir, "accuracies.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(accuracies, f)

    save_path = os.path.join(log_dir, "backdoor_logs.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(backdoor_logs, f)


def extract_celeba(args):
    celeba_dropout_features, celeba_bottleneck_features = utils.extract_celeba_features(args)

@torch.no_grad()
def pca_test(args):
    # Open lfw features
    lfw_dropout_features, _ = utils.extract_lfw_features(args)

    # Try with weight matrix
    W, W_bias = utils.get_weight_matrix(args.model_name)

    bottleneck_features = utils.apply_weight_matrix(W, lfw_dropout_features, bias=W_bias)

    exp_var, eig_vecs = utils.pca(bottleneck_features)

    # Plot explained variance
    plt.figure()
    plt.plot(exp_var)
    plt.yscale("log")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance")
    plt.title(f"PCA for {args.model_name}")
    plt.savefig(f"figures/{args.model_name}_pca.png")

    # Save a csv of explained variance
    csv_name = f"csv/{args.model_name}_pca.csv"
    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "explained_variance"])
        for i, var in enumerate(exp_var):
            writer.writerow([i, var.item()])

    # Do normalised PCA
    # Open lfw features
    lfw_dropout_features, _ = utils.extract_lfw_features(args)

    # Try with weight matrix
    W, W_bias = utils.get_weight_matrix(args.model_name)
    W = utils.normalise_weight_matrix(W)

    bottleneck_features = utils.apply_weight_matrix(W, lfw_dropout_features, bias=W_bias)

    exp_var, eig_vecs = utils.pca(bottleneck_features)

    # Plot explained variance
    plt.figure()
    plt.plot(exp_var)
    plt.yscale("log")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance")
    plt.title(f"PCA for {args.model_name}")
    plt.savefig(f"figures/{args.model_name}_pca_normalised.png")

    # Save a csv of explained variance
    csv_name = f"csv/{args.model_name}_pca_normalised.csv"
    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "explained_variance"])
        for i, var in enumerate(exp_var):
            writer.writerow([i, var.item()])


def get_eigenvalues(args):
    csv_name = f"csv/{args.model_name}_singular_values.csv"

    # Try with weight matrix
    W, W_bias = utils.get_weight_matrix(args.model_name)

    u, s, v = torch.svd(W)
    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Singular Value"])
        for i, s in enumerate(s):
            writer.writerow([i+1, s.item()])

def get_normalised_eigenvalues(args):
    csv_name = f"csv/{args.model_name}_normalised_singular_values.csv"

    # Try with weight matrix
    W, W_bias = utils.get_weight_matrix(args.model_name)
    W = utils.normalise_weight_matrix(W)

    u, s, v = torch.svd(W)
    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Singular Value"])
        for i, s in enumerate(s):
            writer.writerow([i+1, s.item()])

def main(args):
    if args.extract_celeba:
        extract_celeba(args)
    elif args.model_summary:
        utils.print_summary(args.model_name)
    elif args.pca_test:
        pca_test(args)
    elif args.get_eigenvalues:
        get_eigenvalues(args)
    elif args.get_normalised_eigenvalues:
        get_normalised_eigenvalues(args)
    elif args.eps_delta:
        utils.eps_delta(args)
    elif args.angle_dist:
        utils.angle_dist(args)
    else:
        accuracy_test(args)

if __name__ == "__main__":
    args = parse_args.make(*sys.argv[1:])

    main(args)
