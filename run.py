
from requests import patch
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from constants import BATCH_SIZE, LEARNING_RATE, NUM_WORKERS

from cats_vs_dogs_dataset import CatsVsDogsDataset
from vision_transformer import VisionTransformer
from training import train

if __name__ == "__main__":

    torch.cuda.empty_cache()

    transformations_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4886, 0.4555, 0.4172), (0.2526, 0.2458, 0.2487))
    ])

    transformations_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4836, 0.4510, 0.4152), (0.2531, 0.2469, 0.2495))
    ])

    transformations_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4869, 0.4542, 0.4151), (0.2498, 0.2421, 0.2453))
    ])

    train_d = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\train",
        transformations=transformations_train)

    test_d = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\test",
        transformations=transformations_test)

    val_d = CatsVsDogsDataset(
        dataset_path=r"D:\\cnn_vs_transformers\\data\\val",
        transformations=transformations_val)

    train_loader = DataLoader(
        train_d,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_d,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_d,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = VisionTransformer(
        img_size=64,
        patch_size=16,
        in_chans=3,
        n_classes=1,
        depth=1,
        embed_dim=768,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.3,
        attn_p=0.3
    )

    criterion = nn.BCEWithLogitsLoss()
    opt = AdamW(model.parameters(), LEARNING_RATE)

    train(model, train_loader, test_loader, criterion, opt)
    torch.cuda.empty_cache()
