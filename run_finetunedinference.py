# 0_0 Default Module
from input_preprocess import tokenizer, scenarios_list
from inference import lwm_inference
from utils import prepare_loaders
from train import finetune
from lwm_model import lwm

import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 0_1 Added Module
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 0_2 Parameter Setting
n_beams = 16  # Beam Prediction일 경우 사용
task = ['Beam Prediction', 'LoS/NLoS Classification', 'Channel Reconstruction', 'Embedding Regression'][3]  # Default: LoS/NLoS Classification
task_type = ["classification", "regression"][1]  # Default: Classification
visualization_method = ["pca", "umap", "tsne"][2]  # Default: TSNE
input_types = ["channel_emb", "raw"]  # Supported input types: + cls_emb
train_ratios = [.5]  # Fraction of data for training [.001, .01, .05, .1, .25, .5, .8]
fine_tuning_status = [None]  # Fine-tuning configurations [None, ["layers.8", "layers.9", "layers.10", "layers.11"], "full"]
test_type = ["backbone", "full"][0]

# 0_3 Scenarios Setting
all_scenarios = scenarios_list()
selected_scenario_names = [all_scenarios[6]]  # 예: city_6_miami

print(f"Selected Scenario: {selected_scenario_names[0]}")

# 0_4 Tokenizer(Pre-processing)
mask=True
preprocessed_data, labels, raw_chs = tokenizer(
    selected_scenario_names,
    bs_idxs=[3],           # BS index 3번 사용
    load_data=True,       # 이미 있으면 True, 처음 실행이면 False
    task=task,
    n_beams = n_beams,
    mask=mask,
    masking_percent = 0.40
)
if isinstance(preprocessed_data, dict):
    all_samples = [torch.tensor(sample[0], dtype=torch.float32)
                   for samples in preprocessed_data.values()
                   for sample in samples]
    preprocessed_data = torch.stack(all_samples, dim=0)  # [N, 33, 32]
    print("✅ Model Structure Re-Arranged.")
subset_data = preprocessed_data[:1000]
subset_labels = labels[:1000]

# Step 1. 모델 base
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = lwm()

# Step 2. 나머지는 finetune() 함수로 모델을 재구성
fine_tuned_model, *_ = finetune(
    base_model=model,
    train_loader=None,
    val_loader=None,
    task_type=task_type,
    input_type=input_types[0],
    num_classes=n_beams if task == 'Beam Prediction' else 2 if task == 'LoS/NLoS Classification' else None,
    output_dim=992,
    use_custom_head=True,
    fine_tune_layers=None,
    optimizer_config=None,
    epochs=0,
    mask=mask,
    device=device,
    task=task
)

# Step 3. 로드
state_dict = torch.load("results/Embedding Regression/1753677739/raw_epoch1_valLoss475.1160_1753677739.pth", map_location=device)
fine_tuned_model.load_state_dict(state_dict)
fine_tuned_model = fine_tuned_model.to(device)
fine_tuned_model.eval()