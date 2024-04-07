import torchvision
from torch.utils.data import DataLoader


def get_dataloader(
    dataset_name: str,
    data_folder="data",
    batch_size=4,
    shuffle=True,
    num_workers=4
) -> DataLoader:
    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(
            root=data_folder,
            download=True
        )
    elif dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root=data_folder, download=True
        )
    elif dataset_name == "FashionMNIST":
        dataset = torchvision.datasets.FashionMNIST(
            root=data_folder, download=True
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader
