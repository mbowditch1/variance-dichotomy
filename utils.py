from deepface.modules import modeling, detection, preprocessing
import math
from torch.linalg import matrix_rank
from typing import Dict, Iterable, Callable
from tqdm import tqdm
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons import image_utils
from keras.models import Model
from lfw import LFWDataset, read_pairs, get_paths, CelebADataset
import pickle
import numpy as np
import os
from sklearn import metrics
import torch
from torch import nn, Tensor

import net
from PIL import Image

LFW_DIR = "lfw/aligned"
LFW_PAIRS_PATH = "pairs.txt"

MODEL_SUMMARY = {
        "Facenet": {"Dropout_name": "Dropout", "Bottleneck_name": "Bottleneck", "Dropout_size": 1792, "Bottleneck_size": 128},

        "Facenet512": {"Dropout_name": "Dropout", "Bottleneck_name": "Bottleneck", "Dropout_size": 1792, "Bottleneck_size": 512},

        "ArcFace": {"Dropout_name": "flatten", "Bottleneck_name": "dense", "Dropout_size": 25088, "Bottleneck_size": 512},

        "adaface": {"Dropout_name": "output_layer.0", "Bottleneck_name": "output_layer.3", "Dropout_size": 25088, "Bottleneck_size": 512}}

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()


    return uu

def merged_class_projection(embeddings1, embeddings2):
    vec1 = torch.mean(embeddings1, 0)
    vec2 = torch.mean(embeddings2, 0)

    vec = vec1 - vec2
    dim = vec.size()[0]

    # Use Gram Schmidt to find projection to subspace orthogonal to feature
    vec = vec / torch.norm(vec)

    # Calculate projection matrix
    a = torch.eye(dim)
    a[:, 0] = vec
    u = gram_schmidt(a)
    s = torch.eye(dim, dim)
    s[:, 0] = torch.zeros(dim)
    u_inv = torch.inverse(u)
    p = torch.matmul(u, s)
    p = torch.matmul(p, u_inv)

    return p

def find_projection(vecs, mean=True):
    # Average of all features
    if mean:
        vec = torch.mean(vecs, 0)
    else:
        vec = vecs

    if isinstance(vec, np.ndarray):
        vec = torch.Tensor(vec)

    dim = vec.shape[0]

    # Use Gram Schmidt to find projection to subspace orthogonal to feature
    vec = vec / torch.norm(vec)

    # Calculate projection matrix
    a = torch.eye(dim)
    a[:, 0] = vec
    u = gram_schmidt(a)
    s = torch.eye(dim, dim)
    s[:, 0] = torch.zeros(dim)
    u_inv = torch.inverse(u)
    p = torch.matmul(u, s)
    p = torch.matmul(p, u_inv)

    return p

def calc_angles(features):
    num_feat = features.size()[0]
    feat_pair1 = features[np.arange(0, num_feat, 2), :]
    feat_pair2 = features[np.arange(1, num_feat, 2), :]
    feat_dist = [
        torch.dot(x, y)/(torch.norm(x)*torch.norm(y)) for x, y in zip(feat_pair1, feat_pair2)
    ]

    return feat_dist

def load_celeba():
    # Download CelebA dataset and make list of classes
    file_path = "celeba/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print("Downloading CelebA dataset")
        os.system("wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip")
        os.system("unzip celeba.zip")
        os.system("rm celeba.zip")

        # Rearrange directory
        print("Rearranging directory")
        rearrange_celeba()
        os.system("rm -rf img_align_celeba")

    sub_folders = [
        name
        for name in os.listdir(file_path)
        if os.path.isdir(os.path.join(file_path, name))
    ]

    celeba = []
    for s in sub_folders:
        curr_path = file_path + s + "/"
        file_names = os.listdir(curr_path)
        file_names = [curr_path + f for f in file_names]
        celeba.append((s, file_names))

    return celeba

def rearrange_celeba():
    # Rearrange CelebA directory for easier class identification
    f = open("identity_CelebA.txt", "r")
    filenames = []
    classes = []
    for x in f:
        # split line on space
        sp = x.split()
        filenames.append(sp[0])
        classes.append(sp[1])

    # Create directory
    for c in classes:
        if not os.path.exists("celeba/" + str(c)):
            os.makedirs("celeba/" + str(c))

    # Change jpg to png
    filenames = [f.replace(".jpg", ".png") for f in filenames]

    for i in tqdm(range(len(filenames))):
        # Directory
        directory = str(classes[i])

        # Parent Directory path
        parent_dir = "celeba"

        # Path
        path = os.path.join(parent_dir, directory)

        # Source path
        source = "data/" + filenames[i]

        if os.path.isfile(source):
            # Destination path
            destination = "celeba/" + directory + "/" + filenames[i]

            # Move file
            os.rename(source, destination)

def load_adaface_model(path="pretrained/adaface_ir101_ms1mv2.ckpt", architecture='ir_101'):
    # load model and pretrained statedict
    model = net.build_model(architecture)
    statedict = torch.load(path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def adaface_to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

# Again, for adaface
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            _ = self.model(x)
        return self._features

def adaface_get_weights(model_name, model):
    layer = MODEL_SUMMARY[model_name]["Bottleneck_name"]
    # Get weights of output_layer.3
    for name, param in model.named_parameters():
        if name == f"{layer}.weight":
            weights = param
        if name == f"{layer}.bias":
            bias = param

    return weights.T, bias

@torch.no_grad()
def extract_adaface_celeba_features(args):
    dropout_dim = MODEL_SUMMARY[args.model_name]["Dropout_size"]
    bottleneck_dim = MODEL_SUMMARY[args.model_name]["Bottleneck_size"]
    dropout_name = MODEL_SUMMARY[args.model_name]["Dropout_name"]
    bottleneck_name = MODEL_SUMMARY[args.model_name]["Bottleneck_name"]

    celeba_paths = load_celeba()
    print(f"Number of classes in CelebA dataset: {len(celeba_paths)}")

    model = load_adaface_model()
    # Print out model summary
    model.eval()
    W, W_bias = adaface_get_weights(args.model_name, model)

    print(f"w_bias: {W.shape}")
    print(f"w: {W_bias.shape}")

    adaface_features = FeatureExtractor(model, layers=[MODEL_SUMMARY["adaface"]["Dropout_name"], MODEL_SUMMARY["adaface"]["Bottleneck_name"]])

    embeddings = {bottleneck_name: [torch.empty((0, bottleneck_dim), dtype=torch.float32) for i in range(len(celeba_paths))], dropout_name: [torch.empty((0, dropout_dim), dtype=torch.float32) for i in range(len(celeba_paths))]}

    # Extract celeba features
    for i, cl in tqdm(enumerate(celeba_paths), total=len(celeba_paths)):
        for x in cl[1]:
            img = Image.open(x).convert('RGB')
            img = img.resize((112,112))

            bgr_tensor_input = adaface_to_input(img)
            model_output = adaface_features(bgr_tensor_input)

            dropout = model_output[MODEL_SUMMARY["adaface"]["Dropout_name"]]
            bottleneck = model_output[MODEL_SUMMARY["adaface"]["Bottleneck_name"]]
            dropout = dropout.view(dropout.size(0), -1)

            embeddings[dropout_name][i] = torch.cat((embeddings[dropout_name][i], dropout), 0)
            embeddings[bottleneck_name][i] = torch.cat((embeddings[bottleneck_name][i], bottleneck), 0)

        print(f"embeddings: {embeddings[dropout_name][i].shape}")
        print(f"embeddings: {embeddings[bottleneck_name][i].shape}")

    # Save features
    print(f"Saving embeddings for {args.model_name}")
    with open(f"embeddings/celeba_{args.model_name}_dropout_features.pkl", "wb") as f:
        pickle.dump(embeddings[dropout_name], f)

    with open(f"embeddings/celeba_{args.model_name}_bottleneck_features.pkl", "wb") as f:
        pickle.dump(embeddings[bottleneck_name], f)

    return embeddings[dropout_name], embeddings[bottleneck_name]

# Need to handle adaface seperately to other models
@torch.no_grad()
def extract_adaface_features(args):
    dropout_dim = MODEL_SUMMARY[args.model_name]["Dropout_size"]
    bottleneck_dim = MODEL_SUMMARY[args.model_name]["Bottleneck_size"]
    dropout_name = MODEL_SUMMARY[args.model_name]["Dropout_name"]
    bottleneck_name = MODEL_SUMMARY[args.model_name]["Bottleneck_name"]

    model = load_adaface_model()
    # Print out model summary
    model.eval()
    W, W_bias = adaface_get_weights(args.model_name, model)

    print(f"w_bias: {W.shape}")
    print(f"w: {W_bias.shape}")

    adaface_features = FeatureExtractor(model, layers=[MODEL_SUMMARY["adaface"]["Dropout_name"], MODEL_SUMMARY["adaface"]["Bottleneck_name"]])

    pairs = read_pairs(LFW_PAIRS_PATH)
    lfw_paths, issame_list = get_paths(LFW_DIR, pairs, "jpg")
    print(f"Number of images in LFW dataset: {len(lfw_paths)}")

    embeddings = {bottleneck_name: torch.empty((0, bottleneck_dim), dtype=torch.float32), dropout_name: torch.empty((0, dropout_dim), dtype=torch.float32)}

    # Extract lfw features
    for i, x in tqdm(enumerate(lfw_paths), total=len(lfw_paths)):
        img = Image.open(x).convert('RGB')
        img = img.resize((112,112))

        bgr_tensor_input = adaface_to_input(img)
        model_output = adaface_features(bgr_tensor_input)

        dropout = model_output[MODEL_SUMMARY["adaface"]["Dropout_name"]]
        bottleneck = model_output[MODEL_SUMMARY["adaface"]["Bottleneck_name"]]
        dropout = dropout.view(dropout.size(0), -1)

        embeddings[dropout_name] = torch.cat((embeddings[dropout_name], dropout), 0)
        embeddings[bottleneck_name] = torch.cat((embeddings[bottleneck_name], bottleneck), 0)

        print(f"embeddings: {embeddings[dropout_name].shape}")
        print(f"embeddings: {embeddings[bottleneck_name].shape}")

    # Save features
    print(f"Saving embeddings for {args.model_name}")
    with open(f"embeddings/{args.model_name}_dropout_features.pkl", "wb") as f:
        pickle.dump(embeddings[dropout_name], f)

    with open(f"embeddings/{args.model_name}_bottleneck_features.pkl", "wb") as f:
        pickle.dump(embeddings[bottleneck_name], f)

    return embeddings[dropout_name], embeddings[bottleneck_name]

def extract_lfw_features(args):
    if os.path.exists(f"embeddings/{args.model_name}_dropout_features.pkl") or os.path.exists(f"embeddings/{args.model_name}_bottleneck_features.pkl"):
        print(f"Features have already been extracted for {args.model_name}")
        with open(f"embeddings/{args.model_name}_dropout_features.pkl", "rb") as f:
            dropout_features = pickle.load(f)

        try:
            with open(f"embeddings/{args.model_name}_bottleneck_features.pkl", "rb") as f:
                bottleneck_features = pickle.load(f)
        except:
            print("No bottleneck features found")
            bottleneck_features = None

        return dropout_features, bottleneck_features

    if args.model_name == "adaface":
        return extract_adaface_features(args)

    dropout_dim = MODEL_SUMMARY[args.model_name]["Dropout_size"]
    bottleneck_dim = MODEL_SUMMARY[args.model_name]["Bottleneck_size"]
    dropout_name = MODEL_SUMMARY[args.model_name]["Dropout_name"]
    bottleneck_name = MODEL_SUMMARY[args.model_name]["Bottleneck_name"]

    model: FacialRecognition = modeling.build_model(model_name=args.model_name)
    target_size = model.input_shape
    model = model.model

    normalization = "base"
    
    pairs = read_pairs(LFW_PAIRS_PATH)
    lfw_paths, issame_list = get_paths(LFW_DIR, pairs, "jpg")
    print(f"Number of images in LFW dataset: {len(lfw_paths)}")

    embeddings = {bottleneck_name: torch.empty((0, bottleneck_dim), dtype=torch.float32), dropout_name: torch.empty((0, dropout_dim), dtype=torch.float32)}

    for layer_name in [dropout_name, bottleneck_name]:
        print(f"Extracting features from {layer_name} layer")
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        # Extract lfw features
        for i, x in tqdm(enumerate(lfw_paths), total=len(lfw_paths)):
            img, _ = image_utils.load_image(x)

            # rgb to bgr
            img = img[:, :, ::-1]

            img = preprocessing.resize_image(
                img=img,
                # thanks to DeepId (!)
                target_size=(target_size[1], target_size[0]),
            )
            img = preprocessing.normalize_input(img=img, normalization=normalization)

            output = intermediate_layer_model.predict(img)
            output = torch.Tensor(output)

            embeddings[layer_name] = torch.cat((embeddings[layer_name], output), 0)

            print(f"embeddings: {embeddings[layer_name].shape}")

    # Save features
    print(f"Saving embeddings for {args.model_name}")
    with open(f"embeddings/{args.model_name}_dropout_features.pkl", "wb") as f:
        pickle.dump(embeddings[dropout_name], f)

    with open(f"embeddings/{args.model_name}_bottleneck_features.pkl", "wb") as f:
        pickle.dump(embeddings[bottleneck_name], f)

    return embeddings[dropout_name], embeddings[bottleneck_name]

def extract_celeba_features(args):
    if os.path.exists(f"embeddings/celeba_{args.model_name}_dropout_features.pkl") and os.path.exists(f"embeddings/celeba_{args.model_name}_bottleneck_features.pkl"):
        print(f"Features have already been extracted for {args.model_name}")
        with open(f"embeddings/celeba_{args.model_name}_dropout_features.pkl", "rb") as f:
            dropout_features = pickle.load(f)

        try:
            with open(f"embeddings/celeba_{args.model_name}_bottleneck_features.pkl", "rb") as f:
                bottleneck_features = pickle.load(f)
        except:
            print("No bottleneck features found")
            bottleneck_features = None

        return dropout_features, bottleneck_features

    if args.model_name == "adaface":
        return extract_adaface_celeba_features(args)

    celeba_paths = load_celeba()
    print(f"Number of classes in CelebA dataset: {len(celeba_paths)}")

    # Change dimensions based on model
    dropout_dim = MODEL_SUMMARY[args.model_name]["Dropout_size"]
    bottleneck_dim = MODEL_SUMMARY[args.model_name]["Bottleneck_size"]
    dropout_name = MODEL_SUMMARY[args.model_name]["Dropout_name"]
    bottleneck_name = MODEL_SUMMARY[args.model_name]["Bottleneck_name"]

    model: FacialRecognition = modeling.build_model(model_name=args.model_name)
    target_size = model.input_shape
    model = model.model

    normalization = "base"
    
    embeddings = {bottleneck_name: [torch.empty((0, bottleneck_dim), dtype=torch.float32) for i in range(len(celeba_paths))], dropout_name: [torch.empty((0, dropout_dim), dtype=torch.float32) for i in range(len(celeba_paths))]}

    for layer_name in [dropout_name, bottleneck_name]:
        print(f"Extracting features from {layer_name} layer")
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        # Extract CelebA features
        for i, cl in tqdm(enumerate(celeba_paths), total=len(celeba_paths)):
            for j, x in enumerate(cl[1]):
                img, _ = image_utils.load_image(x)

                # rgb to bgr
                img = img[:, :, ::-1]

                img = preprocessing.resize_image(
                    img=img,
                    # thanks to DeepId (!)
                    target_size=(target_size[1], target_size[0]),
                )
                img = preprocessing.normalize_input(img=img, normalization=normalization)

                output = intermediate_layer_model.predict(img)
                output = torch.Tensor(output)

                embeddings[layer_name][i] = torch.cat((embeddings[layer_name][i], output), 0)

            print(f"embeddings: {embeddings[layer_name][i].shape}")

        # Save features
        print(f"Saving celeba embeddings for {args.model_name}")
        with open(f"embeddings/celeba_{args.model_name}_dropout_features.pkl", "wb") as f:
            pickle.dump(embeddings[dropout_name], f)

    with open(f"embeddings/celeba_{args.model_name}_bottleneck_features.pkl", "wb") as f:
        pickle.dump(embeddings[bottleneck_name], f)

    return embeddings[dropout_name], embeddings[bottleneck_name]

def print_summary(model_name):
    model: FacialRecognition = modeling.build_model(model_name=model_name)
    model.model.summary()

def get_weight_matrix(model_name):
    if model_name == "adaface":
        model = load_adaface_model()
        return adaface_get_weights(model_name, model)

    model: FacialRecognition = modeling.build_model(model_name=model_name)
    model = model.model
    W = model.get_layer(MODEL_SUMMARY[model_name]["Bottleneck_name"]).get_weights()[0]
    print(f"Weight matrix shape: {W.shape}")
    W = torch.Tensor(W)

    return W, None

def apply_weight_matrix(W, features, bias=None):
    # Apply weight matrix to features
    weighted_features = torch.matmul(W.T, features.T).T

    # Add bias to each feature
    if bias is not None:
        for i in range(weighted_features.size()[0]):
            weighted_features[i] += bias

    return weighted_features

def calc_accuracy(features, issame_list, angles=None, threshold=0.39):
    if angles is None:
        angles = calc_angles(features)

    predicted = [True if s > threshold else False for s in angles]
    issame_list = np.asarray(issame_list)

    test_accuracy = metrics.accuracy_score(issame_list, predicted)

    return test_accuracy

def train_test_split(input, split, double=False):
    # Splits LFW dataset into training and testing sets
    if double:
        num_in_split = 1200
    else:
        num_in_split = 600

    split_indices = list(range(10))
    split_indices.remove(split)

    test_list = input[split*num_in_split:(split+1)*num_in_split]
    train_list = [input[i*num_in_split:(i+1)*num_in_split] for i in split_indices]
    train_list = [item for sublist in train_list for item in sublist]

    if type(input) == torch.Tensor:
        train_list =  torch.stack(train_list)

    return train_list, test_list

def train_threshold(embeddings, train_issame_list):
    best_acc = 0
    best_t = 0

    print("Training threshold...")
    angles = calc_angles(embeddings)
    for t in np.arange(0, 1, 0.001):
        curr_acc = calc_accuracy(embeddings, train_issame_list, angles=angles, threshold=t)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_t = t

    print(f"Best threshold: {best_t}, Best accuracy: {best_acc}")
    return best_t


def find_best_celeba(dropout_embeddings, bottleneck_embeddings, threshold):
    # Restrict to classes with more than 20 images
    bottleneck_embeddings = [e for e in bottleneck_embeddings if e.shape[0] >= 20]
    dropout_embeddings = [e for e in dropout_embeddings if e.shape[0] >= 20]
    print(f"Number of classes with more than 20 images: {len(dropout_embeddings)}")

    accuracies = []

    for celeb in bottleneck_embeddings:
        accuracies.append(celeba_test_accuracy(celeb, threshold))

    print(f"Mean accuracy: {np.mean(accuracies)}")
    print(f"Max accuracy: {np.max(accuracies)}")
    print(f"Min accuracy: {np.min(accuracies)}")

    acc_2 = accuracies
    acc_2.sort(reverse=True)

    # Sort embeddings based on accuracies
    dropout_embeddings = [x for _, x in sorted(zip(accuracies, dropout_embeddings), key=lambda pair: pair[0], reverse=True)]

    return dropout_embeddings

def celeba_test_accuracy(embeddings, threshold):
    # Get angle between every pair of images
    feat_dist = [
        torch.dot(embeddings[i], embeddings[j])/(torch.norm(embeddings[i])*torch.norm(embeddings[j])) for i in range(embeddings.shape[0]) for j in range(embeddings.shape[0]) if i > j
    ]

    # Calculate accuracy
    test_acc = sum([1 if s > threshold else 0 for s in feat_dist])/len(feat_dist)

    return test_acc

def celeba_test_accuracy_diff_classes(embeddings1, embeddings2, threshold):
    # Get angle between every pair of images
    feat_dist = [
        torch.dot(embeddings1[i], embeddings2[j])/(torch.norm(embeddings1[i])*torch.norm(embeddings2[j])) for i in range(embeddings1.shape[0]) for j in range(embeddings2.shape[0]) if i >= j
    ]

    # Calculate accuracy
    test_acc = sum([1 if s > threshold else 0 for s in feat_dist])/len(feat_dist)

    return test_acc

def add_backdoor_log(backdoor_type, indices):
    return {"backdoor_type": backdoor_type, "indices": indices, "accuracies": []}

def reinflate_weight_matrix(W):
    d = W.shape[0]

    print(f"Rank of W before: {matrix_rank(W)}")

    # Perform SVD on W_0
    u, s, v = torch.svd(W)
    print(f"u shape: {u.shape}")
    print(f"s shape: {s.shape}")
    print(f"v shape: {v.shape}")

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(s.cpu().numpy()[:, None])

    # Move v_d+1 to v_d
    v[d-1] = v[d]

    # Pull sigma_d from KDE
    print(f"Old sigma_d: {s[d-1]}")
    new_s_d = kde_dist.sample()
    # Convert back to tensor
    new_s_d = torch.tensor(new_s_d).to(device)
    s[d-1] = new_s_d

    # Reconstruct W_0
    W = torch.matmul(u, torch.matmul(torch.diag(s), v.T))

    print(f"Rank of W after: {matrix_rank(W)}")

    return W

def normalise_weight_matrix(W):
    print(f"Normalising weight matrix")

    d = W.shape[0]

    # Perform SVD on W_0
    u, s, v = torch.svd(W)
    print(f"u shape: {u.shape}")
    print(f"s shape: {s.shape}")
    print(f"v shape: {v.shape}")

    # s values all the same 
    new_s = torch.mean(s) * torch.ones_like(s)
    s = new_s

    # Reconstruct W_0
    W = torch.matmul(u, torch.matmul(torch.diag(s), v.T))

    return W

def pca(embeddings):
    # If Tensor convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # PCA without mean adjustment
    embeddings = [f / np.linalg.norm(f) for f in embeddings]

    feature_vectors = np.array(embeddings)
    standardized_data = (
        feature_vectors  
    ) / feature_vectors.std(axis=0)

    print("Applying PCA")
    covariance_matrix = np.cov(standardized_data, ddof = 0, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1]

    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
    sorted_eigenvectors = sorted_eigenvectors.T

    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

    return explained_variance, sorted_eigenvectors

def get_pca_matrix(args, embeddings):
    # Get normalising matrix for modified_celeba method
    K_save_path = "embeddings/K_" + args.model_name + ".pkl"
    if not os.path.exists(K_save_path):
        K = find_pca_matrix(embeddings)
        with open(K_save_path, "wb") as f:
            pickle.dump(K, f)

    with open(K_save_path, "rb") as f:
        K = pickle.load(f)

    return K

def find_pca_matrix(embeddings):
    print("Finding PCA matrix")

    # Find the normalising matrix in the modified celeba method
    dim = len(embeddings[0])
    print(f"Dimension of embeddings: {dim}")

    cov = torch.cov(embeddings.T)
    D, U = torch.linalg.eigh(cov)  # Computes UDU^T
    D = torch.sqrt(D)
    D = torch.diag(D)
    for i in range(dim):
        if D[i][i] < 1e-7:
            D[i][i] = 0

    K = torch.matmul(U, D)

    return K

def pca_normalise_weight_matrix(args, W, embeddings, W_bias=None):
    exp_var, eig_vecs = pca(embeddings)
    K = get_pca_matrix(args, embeddings)

    print(f"Old W shape: {W.shape}")
    # Multiply by K inverse
    W = torch.linalg.solve(K, W.T).T
    print(f"New W shape: {W.shape}")
    if W_bias is not None:
        W_bias = torch.linalg.solve(K, W_bias.T).T

    return W, W_bias

@torch.no_grad()
def eps_delta(args):
    # Open lfw features
    lfw_dropout_features, _ = extract_lfw_features(args)

    # Try with weight matrix
    W, W_bias = get_weight_matrix(args.model_name)
    if args.normalise:
        W = normalise_weight_matrix(W)
    lfw_bottleneck_features = apply_weight_matrix(W, lfw_dropout_features, bias=W_bias)

    celeba_dropout_features, _ = extract_celeba_features(args)
    celeba_bottleneck_features = []
    for d_f in celeba_dropout_features:
        celeba_bottleneck_features.append(apply_weight_matrix(W, d_f, bias=W_bias))

    # Stack celeba features
    celeba_bottleneck_features = torch.cat(celeba_bottleneck_features, dim=0)

    print(f"lfw shape: {lfw_bottleneck_features.shape}")
    print(f"celeba shape: {celeba_bottleneck_features.shape}")

    # Calculate explained variances
    print("Calculating explained variances")
    lfw_exp_var, _ = pca(lfw_bottleneck_features)
    celeba_exp_var, _ = pca(celeba_bottleneck_features)

    both_exp_var = [lfw_exp_var, celeba_exp_var]
    dataset_title = ["lfw", "celeba"]
    for exp_var, title in zip(both_exp_var, dataset_title):
        print(f"Running for {title}")
        eps = 1
        delta = 0
        dim = lfw_bottleneck_features.shape[1]
        pc_i = 0
        pc_j = 0
        pc_i_val = 0
        pc_j_val = 0
        for i in range(dim-1):
            for j in range(i+1, dim):
                cur_delta = abs(math.log(exp_var[j]) - math.log(exp_var[i]))/abs(math.log(exp_var[0]) - math.log(exp_var[dim-1]))
                if cur_delta >= 0.5:
                    cur_eps = abs(i-j)/dim
                    if cur_eps < eps:
                        eps = cur_eps
                        delta = cur_delta
                        pc_i = i
                        pc_j = j
                        pc_i_val = math.log(exp_var[i])
                        pc_j_val = math.log(exp_var[j])

        print(f"{args.model_name} & {eps} & {delta} & {pc_i} & {pc_j} \\")
        print()

@torch.no_grad()
def angle_dist(args):
    for model_name in ["Facenet", "Facenet512", "adaface", "ArcFace"]:
        args.model_name = model_name
        pairs = read_pairs(LFW_PAIRS_PATH)
        lfw_paths, issame_list = get_paths(LFW_DIR, pairs, "jpg")

        # Open lfw features
        lfw_dropout_features, _ = extract_lfw_features(args)

        # Try with weight matrix
        W, W_bias = get_weight_matrix(args.model_name)
        if args.normalise:
            W = normalise_weight_matrix(W)

        bottleneck_features = apply_weight_matrix(W, lfw_dropout_features, bias=W_bias)

        # Find distances for matched and mismatched pairs
        angles = calc_angles(bottleneck_features)

        matched = [1-angles[i] for i in range(len(angles)) if issame_list[i]]
        mismatched = [1-angles[i] for i in range(len(angles)) if not issame_list[i]]

        matched_mean = np.mean(matched)
        mismatched_mean = np.mean(mismatched)
        matched_std = np.std(matched)
        mismatched_std = np.std(mismatched)

        with open(f"embeddings/{args.model_name}_thresholds.pkl", "rb") as f:
            thresholds = pickle.load(f)

        threshold = np.mean(thresholds)

        print("-------------------")
        print(f"{args.model_name} matched")
        print(f"({matched_mean}, {matched_std})")
        print(f"{args.model_name} mismatched")
        print(f"({mismatched_mean}, {mismatched_std})")
        print(f"{args.model_name} threshold")
        print(f"{threshold}")
        print("-------------------")
