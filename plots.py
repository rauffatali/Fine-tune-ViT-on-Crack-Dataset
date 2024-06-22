import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

def show_df(
    df: pd.DataFrame, 
    n_samples: int, 
    cols: int = None, 
    figsize: tuple = (10, 10)) -> None:

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

def show_dl(
    dl: DataLoader, 
    n_samples: int = None, 
    cols: int = None, 
    figsize: tuple = (10, 10)) -> None:

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

def plot_metrics_curve(
    train_hist: list, 
    plot: bool = True, 
    save: bool = False, 
    save_dir: Path = None, 
    figsize: tuple = (15, 10)) -> None:

    """
    Plots accuracy, loss, precision, and recall curves for a training history.

    Args:
        train_hist (list): Training history containing loss and accuracy metrics.
        plot (bool, optional): Whether to display the plots (default: True).
        save (bool, optional): Whether to save the plots (default: False).
        save_dir (Path, optional): Directory to save the plots (default: None).
        figsize (tuple, optional): Size of the figure (default: (15, 5)).
    """

    fig = plt.figure(figsize=figsize)

    metrics = ['loss', 'accuracy', 'precision', 'recall']

    for i, m in enumerate(metrics):
        fig.add_subplot(2, 2, i+1)
        plt.title(f"{m.capitalize()} curve")
        plt.xlabel("Epochs")
        plt.ylabel(f"{m.capitalize()}")
        plt.plot(train_hist[f'train_{m}'], label="train")
        plt.plot(train_hist[f'val_{m}'], label="val")
        plt.legend()

    plt.subplots_adjust(hspace=0.25)

    if save:
        if save_dir is None:
            raise ValueError("save_dir argument must be provided when save is True")
        fig.savefig(f'{save_dir}/metrics_curve.png')
        plt.close(fig)

    if plot:
        plt.show()

def plot_conf_matrix(
    y_true: list, 
    y_pred: list, 
    classes: list = None, 
    plot: bool = True, 
    save: bool = False, 
    save_dir: Path = None, 
    figsize: tuple = (10, 7)) -> None:

    """
    Plots and optionally saves a confusion matrix.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        classes: List of class names (default: None).
        plot: Boolean, whether to display the plot (default: True).
        save: Boolean, whether to save the plot (default: False).
        save_dir: Path object specifying the directory to save the plot (default: None).
        figsize: Tuple specifying the figure size (default: (10, 5)).
    """
    if classes is None:
        classes = np.unique(y_true)

    conf_mat = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(classes)) + 0.5, labels=classes)
    ax.set_yticks(np.arange(len(classes)) + 0.5, labels=classes)

    if save:
        if save_dir is None:
            raise ValueError("save_dir argument must be provided when save is True")
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        plt.close(fig)
    
    if plot:
        plt.show()

def plot_roc_curve(
    y_true: list, 
    y_score: list, 
    curves=('micro', 'macro', 'each_class'),
    class_names: list = None,
    plot: bool = True, 
    save: bool = False, 
    save_dir: Path = None,
    figsize: tuple = (10, 7)) -> None:

    """
    Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true: Ground truth (correct) target values.
        y_score: Prediction probabilities for each class returned by a classifier.
        curves: A listing of which curves should be plotted on the resulting plot. 
                Defaults to `("micro", "macro", "each_class")`.
        class_names: List of class names (default: None).
        plot: Boolean, whether to display the plot (default: True).
        save: Boolean, whether to save the plot (default: False).
        save_dir: Path object specifying the directory to save the plot (default: None).
        figsize: Tuple specifying the figure size (default: (10, 5)).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if 'micro' not in curves and 'macro' not in curves and 'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro", "macro", or "each_class"')

    classes = np.unique(y_true)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i], pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc[micro_key] = auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(len(classes))]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)

    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = auc(fpr[macro_key], tpr[macro_key])

    fig = plt.figure(figsize=figsize)
    plt.title('ROC Curves', fontsize="large")

    if 'each_class' in curves:
        colors = ['tab:red', 'tab:blue']
        for i in range(len(classes)):
            cls_name = class_names[i] if class_names else classes[i]
            plt.plot(fpr[i], tpr[i], lw=2, color=colors[i],
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(cls_name, roc_auc[i]))

    if 'micro' in curves:
        plt.plot(fpr[micro_key], tpr[micro_key],
                 label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc[micro_key]),
                 color='tab:green', linestyle='dotted', linewidth=2)

    if 'macro' in curves:
        plt.plot(fpr[macro_key], tpr[macro_key], 
                 label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc[macro_key]),
                 color='tab:orange', linestyle='dotted', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize='medium')
    plt.ylabel('True Positive Rate', fontsize='medium')
    plt.tick_params(labelsize='medium')
    plt.legend(loc='lower right', fontsize='medium')

    if save:
        if save_dir is None:
            raise ValueError("save_dir argument must be provided when save is True")
        fig.savefig(f'{save_dir}/roc_curve.png')
        plt.close(fig)

    if plot: 
        plt.show()

def plot_precision_recall_curve(
    y_true: list, 
    y_score: list,
    curves=('micro', 'each_class'),
    class_names: list = None,
    plot: bool = True,
    save: bool = False, 
    save_dir: Path = None,
    figsize: tuple = (10, 7)) -> None:

    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true: Ground truth (correct) target values.
        y_probas: Prediction probabilities for each class returned by a classifier.
        curves: A listing of which curves should be plotted on the resulting plot. 
                Defaults to `("micro", "each_class")`.
        class_names: List of class names (default: None).
        plot: Boolean, whether to display the plot (default: True).
        save: Boolean, whether to save the plot (default: False).
        save_dir: Path object specifying the directory to save the plot (default: None).
        figsize: Tuple specifying the figure size (default: (10, 5)).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    classes = np.unique(y_true)

    if 'micro' not in curves and 'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it only takes "micro" or "each_class"')

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_true, y_score[:, i], pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    average_precision[micro_key] = average_precision_score(y_true, y_score, average='micro')
    
    fig = plt.figure(figsize=figsize)

    plt.title('Precision-Recall Curve', fontsize="large")

    if 'each_class' in curves:
        colors = ['tab:red', 'tab:blue']
        for i in range(len(classes)):
            cls_name = class_names[i] if class_names else classes[i]
            plt.plot(recall[i], precision[i], lw=2, color=colors[i],
                     label='Precision-recall curve of class {0} (area = {1:0.3f})'.format(cls_name, average_precision[i]))

    if 'micro' in curves:
        plt.plot(recall[micro_key], precision[micro_key],
                 label='micro-average Precision-recall curve (area = {0:0.3f})'.format(average_precision[micro_key]),
                 color='tab:green', linestyle=':', linewidth=4)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tick_params(labelsize="medium")
    plt.legend(loc='best', fontsize="medium")

    if save:
        if save_dir is None:
            raise ValueError("save_dir argument must be provided when save is True")
        fig.savefig(f'{save_dir}/pr_curve.png')
        plt.close(fig)

    if plot: 
        plt.show()

if __name__ == "__main__":
    n_samples = 26
    cols = round(math.sqrt(n_samples))
    print(cols)