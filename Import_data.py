from urllib.request import urlretrieve
import os

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display, Markdown, Latex

# Baseline Imports
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor

if torch.cuda.is_available():
    print("Cuda is available")


# Stanford databases:
# High quality and full datasets

urls = ["https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/PI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NRTI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NNRTI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/INI_DataSet.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/PI_DataSet.Full.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NRTI_DataSet.Full.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/NNRTI_DataSet.Full.txt",
        "https://hivdb.stanford.edu/_wrapper/download/GenoPhenoDatasets/INI_DataSet.Full.txt"]

data_storage = r".\data"

#importing hq urls
for url in urls:
    filename = url.split("/")[-1]

    urlretrieve(url, data_storage + "\\" + filename)

