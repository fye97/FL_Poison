import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datapreprocessor.cinic10 import CINIC10
from datapreprocessor.chmnist import CHMNIST
from plot_utils import plot_label_distribution
from datapreprocessor.tinyimagenet import TinyImageNet
from datapreprocessor.har import HAR


def load_data(args):
    # load dataset
    data_directory = './data'
    if args.dataset == "HAR":
        download = getattr(args, "download", False)
        train_dataset = HAR(
            root=data_directory, train=True, download=download, normalize=True)
        test_dataset = HAR(
            root=data_directory, train=False, download=download, normalize=True)
    elif args.dataset == "EMNIST":
        trans, test_trans = get_transform(args)
        train_dataset = datasets.EMNIST(data_directory, split="digits", train=True, download=True,
                                        transform=trans)
        test_dataset = datasets.EMNIST(
            data_directory, split="digits", train=False, transform=test_trans)
    elif args.dataset in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]:
        trans, test_trans = get_transform(args)
        train_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=True,
                                                         download=True, transform=trans)
        test_dataset = eval(f"datasets.{args.dataset}")(root=data_directory, train=False,
                                                        download=True, transform=test_trans)
    elif args.dataset in ["CHMNIST", "CINIC10", "TinyImageNet"]:
        """
        dataset in custom datasets, such as CHMNIST, CINIC10, TinyImageNet
        """
        trans, test_trans = get_transform(args)
        train_dataset = eval(args.dataset)(root=data_directory, train=True, download=True,
                                transform=trans)
        test_dataset = eval(args.dataset)(root=data_directory, train=False, download=True,
                               transform=test_trans)
    else:
        raise ValueError("Dataset not implemented yet")

    # deal with CIFAR10 list-type targets. CIFAR10 data is numpy array defaultly.
    train_dataset.targets = list_to_tensor(train_dataset.targets)
    test_dataset.targets = list_to_tensor(test_dataset.targets)
    return train_dataset, test_dataset


def list_to_tensor(vector):
    """
    check whether a instance is tensor, convert it to tensor if it is a list.
    """
    if isinstance(vector, list):
        vector = torch.tensor(vector)
    return vector


def subset_by_idx(args, dataset, indices, train=True):
    if args.dataset == "HAR":
        trans = None
    else:
        trans = get_transform(args)[0] if train else get_transform(args)[1]
    dataset = Partition(
        dataset, indices, transform=trans)
    return dataset


def get_transform(args):
    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST", "FEMNIST"] and args.model in ['lenet', 'lenet_bn', "lr"]:
        # resize MNIST to 32x32 for LeNet5
        train_tran = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(args.mean, args.std)])
        test_trans = train_tran
        # define the image dimensions for self.args, so that others can use it, such as DeepSight, lr model
        args.num_dims = 32
    elif args.dataset in ["CINIC10"]:
        train_tran = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)])
        test_trans = train_tran
    elif args.dataset in ["CHMNIST"]:
        if not hasattr(args, "num_dims") or args.num_dims is None:
            args.num_dims = 150
        train_tran = transforms.Compose(
            [
                transforms.Resize((args.num_dims, args.num_dims)),
                transforms.ToTensor(),
                transforms.Normalize(mean=args.mean, std=args.std),
            ]
        )
        test_trans = train_tran
    elif args.dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        args.num_dims = 32 if args.dataset in ['CIFAR10', 'CIFAR100'] else 64
        # data augmentation for small natural images
        train_tran = transforms.Compose([
            transforms.RandomCrop(args.num_dims, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
    elif args.dataset in ["HAR"]:
        # Non-image dataset: no torchvision transform.
        train_tran, test_trans = None, None
    else:
        raise ValueError("Dataset not implemented yet")

    return train_tran, test_trans


def split_dataset(args, train_dataset, test_dataset):
    # agrs.cache_partition: True, False, non-iid, iid, class-imbalanced-iid
    cache_flag = (args.cache_partition ==
                  True or args.cache_partition == args.distribution)
    if cache_flag:
        # ready for cache usage
        # check if the indices are already generated in running_caches folder
        cache_exist, file_path = check_partition_cache(args)
        if cache_exist:
            args.logger.info("Target indices caches to save time")
            with open(file_path, 'rb') as f:
                client_indices = pickle.load(f)
            try:
                lengths = [len(x) for x in client_indices]
                if min(lengths) > 0:
                    return client_indices, test_dataset
                args.logger.warning(
                    f"Cached partition contains empty clients (min_len={min(lengths)}). Regenerating.")
            except Exception:
                args.logger.warning("Cached partition invalid. Regenerating.")

    args.logger.info("Generating new indices")
    if args.distribution in ['iid', 'class-imbalanced_iid']:
        client_indices = iid_partition(args, train_dataset)
        args.logger.info("Doing iid partition")
        if "class-imbalanced" in args.distribution:
            args.logger.info("Doing class-imbalanced iid partition")
            # class-imbalanced iid partition
            for i in range(args.num_clients):
                class_indices = client_class_indices(
                    client_indices[i], train_dataset)
                client_indices[i] = class_imbalanced_partition(
                    class_indices, args.im_iid_gamma)
    elif args.distribution in ['non-iid']:
        # dirichlet partition
        args.logger.info("Doing non-iid partition")
        client_indices = dirichlet_split_noniid(
            train_dataset.targets, args.dirichlet_alpha, args.num_clients)
        args.logger.info(f"dirichlet alpha: {args.dirichlet_alpha}")
    if cache_flag:
        save_partition_cache(client_indices, file_path)
        # if "class-imbalanced" in args.distribution:
        #     # class-imbalanced partition for test dataset for evaluation
        #     test_class_indices = dataset_class_indices(test_dataset)
        #     test_class_indices = class_imbalanced_partition(
        #         test_class_indices, args.im_iid_gamma)
        #     test_dataset = subset_by_idx(
        #         args, test_dataset, test_class_indices)
    args.logger.info(f"{args.distribution} partition finished")
    # plot the visualization of label distribution of the clients
    # plot_label_distribution(train_dataset, client_indices, args.num_clients, args.dataset, args.distribution)
    return client_indices, test_dataset


def save_partition_cache(client_indices, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(client_indices, f)


def check_partition_cache(args):
    cache_exist = None
    folder_path = 'running_caches'
    file_name = f'{args.dataset}_balanced_{args.distribution}_{args.num_clients}_indices'
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        cache_exist = True if os.path.exists(file_path) else False
    return cache_exist, file_path


def check_noniid_labels(args, train_dataset, client_indices):
    """
    check the unique labels of each client and the common labels across all clients
    """
    client_unique_labels = {}
    common_labels = None
    for client_id, indices in enumerate(client_indices):
        # get the labels of the corresponding indices
        labels = train_dataset.targets[indices]
        # get the unique labels of the client
        unique_labels = set(labels.tolist())
        client_unique_labels[client_id] = unique_labels
        # for the first client, initialize common_labels as the unique labels
        if common_labels is None:
            common_labels = unique_labels
        else:
            # update common_labels by taking the intersection of the unique labels
            common_labels = common_labels.intersection(unique_labels)

    # log the unique labels of each client and the common labels across all clients
    args.logger.info(
        f"Common unique labels across all clients: {common_labels}")
    for client_id, unique_labels in client_unique_labels.items():
        args.logger.info(
            f"Client {client_id} has unique labels: {unique_labels}")


class Partition(Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.classes = getattr(dataset, "classes", None)
        raw_indices = indices if indices is not None else range(len(dataset))
        if torch.is_tensor(raw_indices):
            self.indices = [int(i) for i in raw_indices.cpu().tolist()]
        elif isinstance(raw_indices, np.ndarray):
            self.indices = [int(i) for i in raw_indices.astype(np.int64).tolist()]
        else:
            self.indices = [int(i) for i in raw_indices]

        # Prefer dataset.data/dataset.targets if available; fall back to common custom dataset attrs.
        self.data = None
        if hasattr(dataset, "data"):
            base_data = dataset.data
            if isinstance(base_data, (list, tuple)):
                self.data = [base_data[i] for i in self.indices]
            else:
                self.data = base_data[self.indices]
        elif hasattr(dataset, "image_paths"):
            base_data = dataset.image_paths
            self.data = [base_data[i] for i in self.indices]

        base_targets = None
        for attr in ("targets", "labels", "train_labels"):
            if hasattr(dataset, attr):
                base_targets = getattr(dataset, attr)
                break
        if base_targets is None:
            self.targets = None
        elif isinstance(base_targets, (list, tuple)):
            self.targets = torch.tensor([base_targets[i] for i in self.indices])
        else:
            self.targets = base_targets[self.indices]

        # mode='L' for MNIST-like grey images; mode='RGB' for color images or image paths.
        self.mode = 'RGB'
        if self.data is not None and not isinstance(self.data, (list, tuple)):
            try:
                self.mode = 'L' if len(self.data.shape) == 3 else 'RGB'
            except Exception:
                self.mode = 'RGB'
        self.transform = transform
        self.poison = False

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.data is None:
            # Fallback for datasets without materialized data storage.
            real_idx = self.indices[idx]
            image, target = self.dataset[real_idx]
            return image, target

        image, target = self.data[idx], self.targets[idx] if self.targets is not None else None

        # Convert to PIL Image for transforms.
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image).convert(self.mode)
        else:
            # Convert image to numpy array. For MNIST-like dataset, image is torch tensor;
            # for CIFAR-like dataset, image type is numpy array.
            if not isinstance(image, (np.ndarray, np.generic)):
                image = image.numpy()
            image = Image.fromarray(image, mode=self.mode)
        if self.transform:
            image = self.transform(image)

        if self.poison:
            target_tensor = target if torch.is_tensor(target) else torch.tensor(target)
            image, target = self.synthesizer.backdoor_batch(
                image, target_tensor.reshape(-1, 1))
        if torch.is_tensor(target):
            target = target.squeeze()
        return image, target

    def poison_setup(self, synthesizer):
        self.poison = True
        self.synthesizer = synthesizer


def iid_partition(args, train_dataset):
    """
    nearly-quantity-balanced and class-balanced IID partition for clients.
    """
    labels = train_dataset.targets
    client_indices = [[] for _ in range(args.num_clients)]
    for cls in range(len(train_dataset.classes)):
        # get the indices of current class
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # get the number of sample class=cls indices for each client
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        # random permutation
        class_indices = class_indices[torch.randperm(len(class_indices))]

        # calculate the number of samples for each client
        num_samples = len(class_indices)
        num_samples_per_client_per_class = num_samples // args.num_clients
        # other remaining samples
        remainder_samples = num_samples % args.num_clients

        # uniformly distribute the samples to each client
        for client_id in range(args.num_clients):
            start_idx = client_id * num_samples_per_client_per_class
            end_idx = start_idx + num_samples_per_client_per_class
            client_indices[client_id].extend(
                class_indices[start_idx:end_idx].tolist())
        # distribute the remaining samples to the first few clients
        for i in range(remainder_samples):
            client_indices[i].append(
                class_indices[-(i + 1)].item())
    client_indices = [torch.tensor(indices) for indices in client_indices]
    return client_indices


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    Function: divide the sample index set into n_clients subsets according to the Dirichlet distribution with parameter alpha
    References:
    [orion-orion/FedAO: A toolbox for federated learning](https://github.com/orion-orion/FedAO)
    [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335)
    '''
    # Convert labels to a 1D numpy array on CPU for stable numpy ops.
    if torch.is_tensor(train_labels):
        labels = train_labels.detach().cpu().numpy()
    else:
        labels = np.asarray(train_labels)
    labels = labels.reshape(-1)

    n_classes = int(labels.max()) + 1
    min_size = 1
    max_retry = 100

    # Retry sampling until every client gets at least `min_size` samples.
    for _ in range(max_retry):
        # (K, N) category label distribution matrix X, recording the proportion of each category assigned to each client
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

        # Record the sample index sets corresponding to N clients
        client_parts = [[] for _ in range(n_clients)]

        for y, fracs in enumerate(label_distribution):
            k_idcs = np.where(labels == y)[0].astype(np.int64, copy=False)
            np.random.shuffle(k_idcs)

            # Split indices for class y according to fracs.
            split_points = (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int)
            for i, part in enumerate(np.split(k_idcs, split_points)):
                client_parts[i].append(part)

        client_idcs = [
            np.concatenate(parts).astype(np.int64, copy=False) if len(parts) else np.array([], dtype=np.int64)
            for parts in client_parts
        ]
        if min(len(idcs) for idcs in client_idcs) >= min_size:
            return [torch.tensor(idcs, dtype=torch.int64) for idcs in client_idcs]

    # Last-resort fix: ensure no empty client by moving 1 sample from the largest client.
    client_idcs = [
        np.concatenate(parts).astype(np.int64, copy=False) if len(parts) else np.array([], dtype=np.int64)
        for parts in client_parts
    ]
    empty = [i for i, idcs in enumerate(client_idcs) if len(idcs) == 0]
    while empty:
        donor = int(np.argmax([len(idcs) for idcs in client_idcs]))
        if len(client_idcs[donor]) <= 1:
            break
        take = int(client_idcs[donor][-1])
        client_idcs[donor] = client_idcs[donor][:-1]
        recv = empty.pop()
        client_idcs[recv] = np.array([take], dtype=np.int64)

    if min(len(idcs) for idcs in client_idcs) < 1:
        raise ValueError("Dirichlet partition produced empty client(s). Try a larger dirichlet_alpha.")

    return [torch.tensor(idcs, dtype=torch.int64) for idcs in client_idcs]


def dataset_class_indices(dataset, class_label=None):
    num_classes = len(dataset.classes)
    if class_label:
        return torch.tensor(np.where(dataset.targets == class_label)[0])
    else:
        class_indices = [torch.tensor(np.where(dataset.targets == i)[
            0]) for i in range(num_classes)]
        return class_indices


def client_class_indices(client_indice, train_dataset):
    """
    Given the a client indice, return the list of indices of each class
    """
    labels = train_dataset.targets
    return [client_indice[labels[client_indice] == cls] for cls in range(len(train_dataset.classes))]


def class_imbalanced_partition(class_indices, im_iid_gamma, method='exponential'):
    """
    Perform exponential sampling on the number of each classes.

    Args:
        class_indices (list): A list of tensor containing index of each class for each client
        gamma (float): The exponential decay rate (0 < gamma <= 1).
        method (str, optional): The sampling method, exponential or step. Default as 'exponential'.

    Returns:
        sampled_class_indices (1d tensor): exponential-sampled class_indices
    """
    num_classes = len(class_indices)
    num_sample_per_class = [max(1, int(im_iid_gamma**(i / (num_classes-1)) * len(class_indices[i])))
                            for i in range(num_classes)]
    sampled_class_indices = [class_indices[i][torch.randperm(
        len(class_indices[i]))[:num_sample_per_class[i]]] for i in range(num_classes)]
    # print(f"num_sample_per_class: {num_sample_per_class}")
    return torch.cat(sampled_class_indices)


if __name__ == "__main__":
    pass
