import torchvision
from torch.utils.data import DataLoader


def get_dataloader(
    dataset_name: str, data_folder="data", batch_size=4, shuffle=True, num_workers=4
) -> DataLoader:
    if dataset_name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=True, download=True
        )

        test_dataset = torchvision.datasets.MNIST(
            root=data_folder, train=False, download=True
        )

    elif dataset_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True
        )
    elif dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=True, download=True
        )

        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, train=False, download=True
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader
