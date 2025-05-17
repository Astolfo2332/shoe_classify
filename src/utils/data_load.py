import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.data_setup import download_and_get_dataset


def load_data(output_path: str, batch_size: int, test: int = 0.3, transform=None, augmentation=True, all_data=False):

    print("[INFO] Cargando datos")
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomRotation(30),
        transforms.RandomVerticalFlip(0.6),
    ])

    dataset_path = download_and_get_dataset()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #Totensor automáticamente normaliza a [0,1]
        transforms.ToTensor(),
        # Realizamos la augmentación solo si se indica
        augmentation_transform if augmentation else transforms.Lambda(lambda x: x),
        # Normalizamos con los datos sugeridos por ResNet https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    targets = np.array(dataset.targets)
    #Separación de test y train en partes iguales para cada clase
    splits = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=23)

    for train_idx, val_idx in splits.split(np.zeros(len(targets)), targets):
        train_indices = Subset(dataset, train_idx)
        val_indices = Subset(dataset, val_idx)

    train_loader = DataLoader(train_indices, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_indices, batch_size=batch_size, shuffle=True)

    print("[INFO] tamaño del train_loader:", len(train_loader.dataset))
    print("[INFO] tamaño del test_loader:", len(test_loader.dataset))

    return train_loader, test_loader

if __name__ == "__main__":
    load_data("output/", 32)