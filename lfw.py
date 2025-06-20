import os
import torch
import collections
import PIL.Image
from torch.utils import data


workers = 0 if os.name == "nt" else 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))


class LFWDataset(data.Dataset):
    """
    Dataset subclass for loading LFW images in PyTorch.
    This returns multiple images in a batch.
    """

    def __init__(self, path_list, issame_list, transforms, split="test"):
        """
        Parameters
        ----------
        path_list    -   List of full path-names to LFW images
        """
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] = path_list
        self.pair_label = issame_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        im_out = self.transforms(img)
        im_out = im_out.to(device)
        return im_out


class CelebADataset(data.Dataset):
    """
    Dataset subclass for loading CelebA images in PyTorch.
    This returns multiple images in a batch.
    """

    def __init__(self, path_list, transforms):
        """
        Parameters
        ----------
        path_list    -   List of full path-names to CelebA images
        """
        self.files = collections.defaultdict(list)
        self.files = path_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_file = self.files[index]
        img = PIL.Image.open(img_file)
        im_out = self.transforms(img)
        im_out = im_out.to(device)
        return im_out


def read_pairs(pairs_filename, lfw_flag=True):
    pairs = []
    with open(pairs_filename, "r") as f:
        if lfw_flag:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        else:
            for line in f.readlines():
                pair = line.strip().split()
                pairs.append(pair)

    return pairs


def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(
                lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[1]) + "." + file_ext
            )
            path1 = os.path.join(
                lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[2]) + "." + file_ext
            )
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(
                lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[1]) + "." + file_ext
            )
            path1 = os.path.join(
                lfw_dir, pair[2], pair[2] + "_" + "%04d" % int(pair[3]) + "." + file_ext
            )
            issame = False
        if os.path.exists(path0) and os.path.exists(
            path1
        ):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print("Skipped %d image pairs" % nrof_skipped_pairs)

    return path_list, issame_list
