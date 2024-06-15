import math
import random
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

def show_df(df: pd.DataFrame, n_samples: int, cols: int = None, figsize: tuple = (10, 10)) -> None:
    """
    Displays a grid of n_samples images and their corresponding labels from the DataFrame.

    Args:
        n_samples: The number of samples to display.
        df: The DataFrame containing 'image' and 'label' columns.
        figsize: The size of the figure (width, height) in inches.
    """
    if n_samples > len(df):
        n_samples = len(df)

    if cols is None:
        cols = round(math.sqrt(n_samples))

    rows, reminder = divmod(n_samples, cols)

    if reminder > 0:  # Adding an extra row
        rows += 1

    if n_samples < cols:
        fig = plt.figure(figsize=figsize)
        for i in range(n_samples):
            fig.add_subplot(rows, cols, (i+1))
            rand_idx = random.randint(0, len(df) - 1)
            sample = df[['image', 'label']].iloc[rand_idx]
            sample_img = Image.open(sample['image']).convert("RGB")
            plt.imshow(sample_img)
            plt.title(f"Label: {sample['label']}")
            plt.margins(x=0, y=0)
            plt.axis('off')
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes_flat = axes.flatten()
        for i in range(n_samples):
            rand_idx = random.randint(0, len(df) - 1)
            sample = df[['image', 'label']].iloc[rand_idx]
            sample_img = Image.open(sample['image']).convert("RGB")
            axes_flat[i].imshow(sample_img)
            axes_flat[i].axis('off')
            axes_flat[i].set_title(f"Label: {sample['label']}")
            axes_flat[i].margins(x=0, y=0)

        for j in range(n_samples, rows*cols):
            axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()

def show_dl(dl: DataLoader, n_samples: int = None, cols: int = None, figsize: tuple = (10, 10)) -> None:

    """
    Displays a grid of n_samples images and their corresponding labels from the Pytorch Dataloader.

    Args:
        dl: The Dataloader containing batches of images and corresponding labels.
        n_samples: The number of samples to display.
        cols: The number of columns in the figure.
        figsize: The size of the figure (width, height) in inches.
    """
    images, labels = next(iter(dl))
    images = images * 255.0  # Scale to 0-255 range
    images = np.clip(images, 0, 255) / 255.0  # Clip to 0-1 range
    
    if n_samples:
        if n_samples > len(images):
            n_samples = len(images)
        
        if cols is None:
            cols = round(math.sqrt(n_samples))

        rows, reminder = divmod(n_samples, cols)
        if reminder > 0:  # Adding an extra row
            rows += 1
    else:
        n_samples = len(images)

        if cols is None:
            cols = round(math.sqrt(n_samples))

        rows, reminder = divmod(n_samples, cols)
        if reminder > 0:  # Adding an extra row
            rows += 1
    
    print(f"Images batch shape: {images.size()}")
    print(f"Labels batch shape: {len(labels)}")

    if (n_samples or len(images)) < cols:
        fig = plt.figure(figsize=figsize)
        
        for i in range(n_samples):
            fig.add_subplot(rows, cols, (i+1))
            plt.imshow(images[i].permute(1, 2, 0))
            plt.title(f"Label: {'cracked' if labels[i]==1 else 'noncracked'}")
            plt.axis('off')
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes_flat = axes.flatten()

        for i in range(n_samples):
            axes_flat[i].imshow(images[i].permute(1, 2, 0))
            axes_flat[i].axis('off')
            axes_flat[i].set_title(f"Label: {'cracked' if labels[i]==1 else 'noncracked'}")

        for j in range(n_samples, rows*cols):
            axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_acc_loss_curve(train_hist: list, save: bool = False, figsize: tuple = (15, 5)) -> None:
    fig = plt.figure(figsize=figsize)
    
    fig.add_subplot(1, 2, 1)
    plt.title("Loss curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(train_hist['train_losses'], label="train")
    plt.plot(train_hist['val_losses'], label="val")
    plt.legend()

    fig.add_subplot(1, 2, 2)
    plt.title("Accuracy curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(train_hist['train_accs'], label="train")
    plt.plot(train_hist['val_accs'], label="val")
    plt.legend()

    if save:
        plt.savefig('acc_loss_curve.png')

    plt.show()

if __name__ == "__main__":
    n_samples = 26
    cols = round(math.sqrt(n_samples))
    print(cols)