import os
import random
import pandas as pd
from pathlib import Path

from utils.data_utils import is_imbalanced, prep_data

def balance_data(data: dict, seed: int = 42, samp_to_add: int = 0) -> dict:
    """
    Balances class representation within datasets stored in a dictionary.

    Args:
        data (dict): A dictionary containing datasets as key-value pairs. 
                    Keys are dataset names, values are dictionaries with class labels as keys 
                    and lists of image paths and labels as values.
        seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        dict: The updated `data` dictionary with balanced datasets.
    """
    random.seed(seed)
    for dataset in data:
        if is_imbalanced(data[dataset]):
            class_counts = {cls: len(data[dataset][cls]) for cls in data[dataset]}
            underrepresented, overrepresented = min(class_counts, key=class_counts.get), max(class_counts, key=class_counts.get)
            data[dataset][overrepresented] = random.choices(
                data[dataset][overrepresented], 
                k=(len(data[dataset][underrepresented]) + samp_to_add)
            )
    
    return data

def get_data_as_dataframe(path: Path, dataset_name: str = None, balance: bool = False) -> pd.DataFrame:
    """
    This function reads data from a folder structure and returns a DataFrame for a dataset.

    The function assumes the following directory structure:
        path/
        dataset_name/
        class_1/
        image1.jpg
        image2.jpg
        ...
        class_2/
        image3.jpg
        image4.jpg
        ...

    Args:
        path (Path): The base path of the folder containing datasets.
        dataset_name (str, Optional): The name of the specific dataset to load (default is None).
        balance (bool, Optional): A boolean flag indicating whether to perform class balancing on the data (default is False).

    Returns:
        A pandas DataFrame containing the loaded data or None if the dataset is not found.

    Raises:
        ValueError: If the folder structure is not as expected.
    """
    data = dict()

    datasets = os.listdir(path)
    
    if dataset_name in datasets:
        try:
            data = prep_data(path, data, dataset_name)
        except:
            raise ValueError('Unknown folder structure.')
    elif dataset_name == 'all':
        try:
            for d_name in datasets:
                data = prep_data(path, data, d_name)
        except:
            raise ValueError('Unknown folder structure.')
    else:
        raise ValueError(f"Improper dataset name: {dataset_name}. Please, use one of {datasets+['all']}")
    
    if balance:
        data = balance_data(data)

    images = []
    labels = []

    for _, class_data in data.items():
        for _, image_data in class_data.items():
            for image, label in image_data:
                images.append(image)
                labels.append(label)

    return pd.DataFrame({'image': images, 'label': labels})