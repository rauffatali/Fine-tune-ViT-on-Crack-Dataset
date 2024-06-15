import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from sklearn.model_selection import train_test_split

def is_imbalanced(data: dict, threshold_ratio: float = 1.2) -> bool:
    """
    Checks for binary class imbalance based on a ratio threshold.

    Args:
        data: A dictionary or list where keys or elements represent class labels.
        threshold_ratio: The ratio between majority and minority class size to consider imbalance (default=1.5).

    Returns:
        True if class imbalance is detected, False otherwise.
    """
    if not data:
        raise ValueError("Data dictionary cannot be empty")
    
    class_counts = {cls: len(data[cls]) for cls in data}
    
    majority_count = max(class_counts.values())
    minority_count = min(class_counts.values())

    if majority_count / minority_count > threshold_ratio:
        return True
    else:
        return False
    
def prep_data(path: Path, data: dict, dataset_name: str = None) -> dict:

    """
    Args:
        path (Path): The path to the directory containing the dataset.
        data (dict): A dictionary to store the preprocessed data. 
        dataset_name (str, optional): The name of the dataset within the `data` dictionary (default is None).

    Returns:
        dict: The updated `data` dictionary with the preprocessed image data.

    """

    data[dataset_name] = dict()
    for cls in os.listdir(os.path.join(path, dataset_name)):
        data_cls_path = os.path.join(path, dataset_name, cls)
        for filename in sorted(os.listdir(data_cls_path)):
            if cls not in data[dataset_name]:
                data[dataset_name][cls] = []
            data[dataset_name][cls].append([os.path.join(data_cls_path, filename), cls])
    
    return data

def calculate_mean_std(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    This function calculates the mean and standard deviation of pixel values across all images in a DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing columns for image paths ('path') and labels ('label').

    Returns:
        tuple: A tuple containing two NumPy arrays: mean and standard deviation of pixel values.
    """

    images = np.array([np.array(Image.open(path).convert('RGB')) / 255.0 for path in df['image']])
    
    # Flatten the image data for calculating mean and std
    data = images.reshape(-1, images.shape[-1])

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

def custom_train_test_split(df: pd.DataFrame, test_size: float = 0.2, stratify=None, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training, validation, and test sets while preserving class distribution (if applicable).

    This function performs stratified train-test splitting on a DataFrame in a single step using scikit-learn's `train_test_split` function. 
    It directly splits the data into three sets based on the provided `test_size`.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be split.
        test_size (float, optional): The proportion of the data for the combined validation and test sets (default is 0.2).
        stratify (array-like, optional): If provided, stratification is applied (maintains class distribution across splits).
        seed (int, optional): Seed for the random splits. Defaults to 42.

    Returns:
        pd.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training DataFrame, validation DataFrame, and test DataFrame.

    Raises:
        ValueError: If `test_size` is not between 0.0 and 1.0.
    """

    if not 0.0 <= test_size <= 1.0:
        raise ValueError("test_size must be between 0.0 and 1.0")

    validation_ratio = test_size / (1 - test_size)
    if stratify:
        train_df, valid_test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[stratify].values)
        valid_df, test_df = train_test_split(valid_test_df, test_size=validation_ratio, random_state=seed, stratify=valid_test_df[stratify].values)
    else:
        train_df, valid_test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
        valid_df, test_df = train_test_split(valid_test_df, test_size=validation_ratio, random_state=seed, stratify=stratify)

    return train_df, valid_df, test_df