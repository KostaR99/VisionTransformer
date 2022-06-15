import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

from constants import BATCH_SIZE, NUM_WORKERS
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


class CatsVsDogsDataset(Dataset):
    def __init__(self, dataset_path: str, transformations):
        dataset = pd.read_csv(dataset_path)
        self.x = dataset["image_path"]
        self.x = self.x.to_numpy()

        self.y = dataset["class"]
        self.y = self.y.to_numpy()
        self.transformations = transformations

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        img_path = self.x[index]
        img_class = self.y[index]

        img = Image.open(img_path)
        img = img.convert("RGB")

        if self.transformations:
            img = self.transformations(img)

        return img, float(img_class)


if __name__ == "__main__":

    transformations_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    transformations_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    transformations_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\train",
        transformations=transformations_train)

    test = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\test",
        transformations=transformations_test)

    val = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\val",
        transformations=transformations_val)

    train_loader = DataLoader(
        train,
        BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val,
        BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test,
        BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    mean_train, std_train = get_mean_and_std(train_loader)
    mean_val, std_val = get_mean_and_std(val_loader)
    mean_test, std_test = get_mean_and_std(test_loader)

    print(f"Train dataset: mean: {mean_train}, std: {std_train}")
    print(f"Val dataset: mean: {mean_val}, std: {std_val}")
    print(f"Test dataset: mean: {mean_test}, std: {std_test}")
