import os
import torch
from tqdm import tqdm

from constants import DEVICE, NUM_EPOCHS

def train_one_epoch(model, train_loader, criterion, optimizer):
    train_losses = []
    loop_train = tqdm(train_loader)

    for _, (x, y) in enumerate(loop_train):

        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        y = y.unsqueeze(1)
        loss = criterion(outputs, y)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()

    avg_train_loss = sum(train_losses) / len(train_losses)

    return avg_train_loss

def validate_one_epoch(model, val_loader, criterion):
    val_losses = []
    loop_val = tqdm(val_loader)
    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(loop_val):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y = y.unsqueeze(1)
            outputs = model(x)
            loss = criterion(outputs, y)
            val_losses.append(loss)

    model.train()
    avg_val_loss = sum(val_losses) / len(val_losses)

    return avg_val_loss


def save_checkpoint(model, checkpoint_name: str):
    print("Saving the chekpoint...")
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("./checkpoints", checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved! Path to checkpoint is: {checkpoint_path} .")


def train(model, train_loader, val_loader, criterion, optimizer):
    train_losses = []
    val_losses = []
    model.to(DEVICE)


    for epoch in range(NUM_EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        avg_val_loss = validate_one_epoch(model, val_loader, criterion)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train loss: {avg_train_loss}, Validation loss: {avg_val_loss}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, checkpoint_name=f"ViT_epoch_{epoch}")

    return train_losses, val_losses
