# Import required libraries
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from openpyxl import load_workbook
import xlsxwriter

Cpt_6 = pd.read_excel('Depth-qt-u2-Vs.xlsx')
Cpt_6 = Cpt_6.apply(pd.to_numeric, errors='coerce')
Cpt_6=Cpt_6.dropna(axis=0,how='any')
Cpt_6

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Cpt_6.drop("Vs", axis=1).values)
y = Cpt_6["Vs"].values

# Set K-fold cross-validation and early stopping parameters
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)
patience = 20
min_delta = 0.0001
n_epochs = 1000

# Set batch size
batch_size = 128