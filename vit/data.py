import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms
from config import config


class SymbolDataset(Dataset):
    def __init__(self, data, transform=None, le=None):
        """
        data -> [[img, label], [img, label]]
        """
        super().__init__()
        labels = [i[1] for i in data]
        if le is None:
            le = {l: i for i, l in enumerate(set(labels))}
        self.le = le
        self.labels = [self.le[i] for i in labels]
        self.data = [i[0] for i in data]
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

    def get_le(self):
        """
        label: index
        """
        return self.le

    def get_reverse_le(self):
        """
        index: label
        """
        reverse_le = {i: j for j, i in self.le.items()}
        return reverse_le

    def get_weights(self):
        freqs = [0] * len(self.le)
        for l in self.labels:
            freqs[l] += 1
        m = max(freqs)
        weights = [m/w for w in freqs]
        return weights


class SymbolDatasetTest(Dataset):
    def __init__(self, data, transform=None):
        """
        data -> [img, img, ...
        """
        super().__init__()
        self.data = [d[0] for d in data]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.open(img).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)


def get_data(root_dir, train_size=0.8, train_batch_size=32, test_batch_size=32, train_transform=None, test_transform=None, shuffle=True):
    data = dir_to_list(root_dir)
    train_items, test_items = random_split_list(data, train_size)
    train_ds = SymbolDataset(train_items, transform=train_transform)
    le = train_ds.get_le()
    reverse_le = train_ds.get_reverse_le()
    train_weights = train_ds.get_weights()
    test_ds = SymbolDataset(test_items, transform=test_transform, le=le)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, reverse_le, train_weights


def get_data_test(root_dir, batch_size=32, transform=None):
    data = dir_to_list(root_dir, test=True)
    ds = SymbolDatasetTest(data, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    return loader, data


def random_split_list(l, split_size):
    """
    l = [[item, label], [item, label] ...
    """
    all_labels = set([f[1] for f in l])
    test_nr = int(len(l)*split_size)
    while True:
        random.shuffle(l)
        test_items, train_items = l[:test_nr], l[test_nr:]
        # check if all labels are represented in train set
        test_labels = set([i[1] for i in test_items])
        train_labels = set([i[1] for i in train_items])
        if train_labels == all_labels:
            break

    return test_items, train_items


def dir_to_list(path,
                extensions=('jpg', 'png', 'jpeg', 'webp'),
                test=False,
                ):
    """
    loads dir and returns list of tuples (path_to_img, group)
    images are assumed to be saved as "group/img.<ext>"
    path -> path to directory
    extensions -> only take images with those extensions
    test -> labels are None
    """
    def get_group(filename):
        return str(filename).split('/')[-2] if not test else None
    # convert to tuple
    extensions = tuple(extensions)

    # find all images
    p = Path(path)
    all_files = [(str(file), get_group(file))
                 for file in p.rglob("*")
                 if str(file).lower().endswith(extensions)]

    return all_files


def get_transform():
    transform = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transform


def get_img_to_tensor():
    transform = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.ToTensor(),
        ])
    return transform
