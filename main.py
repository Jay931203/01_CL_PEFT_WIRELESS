# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:24:21 2024

This script pre-trains the LWM model

@author: salikha4
"""
import torch
import torch.nn as nn
from torch.utils.data import random_split
from input_preprocess import tokenizer, scenarios_list
from utils import create_dataloader, count_parameters
import numpy as np
import lwm_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from train import train_lwm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#%% SETTINGS
EPOCHS = 50
BATCH_SIZE = 128 
VAL_BATCH_SIZE = 64 
WARMUP_EPOCHS = 5
BASE_LR = 5e-4
N_ROWS = 4
N_COLUMNS = 4
ELEMENT_LENGTH = N_ROWS*N_COLUMNS*2
D_MODEL = 128 
MAX_LEN = 513
N_LAYERS = 12 
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.999
MASK_PERCENT = 0.40
N_HEADS = 8
DROPOUT = 0.1
#%% GENERATE DATASET
bs_idxs = [1, 2, 3] 
selected_scenario_names = scenarios_list()[:80] 
preprocessed_data = tokenizer(
    selected_scenario_names, 
    MAX_LEN, 
    masking_percent=MASK_PERCENT, 
    mask=True, 
    seed=42
) 
#%% SPLIT DATASET
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
train_ratio = 0.8
val_ratio = 0.2
train_data = {}
val_data = {}
test_data = {}
for key, samples in preprocessed_data.items():
    print(f"key: {key}")
    total_samples = len(samples)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - val_size - train_size
    
    train_data[key], val_data[key], test_data[key] = random_split(
        samples, [train_size, val_size, test_size]
    )
train_loaders = create_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loaders = create_dataloader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)
#%% INITIALIZE MODEL
load_model = True
gpu_ids = [0]
device = torch.device("cuda:0")
model = lwm_model.lwm().to(device)

if load_model:
    model_name = "lwm_epoch50_train0.0077_val0.0060_masking0.40.pth"
    state_dict = torch.load(f"models/{model_name}", map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
model = nn.DataParallel(model, gpu_ids)
print(f"Model loaded successfully on GPU {device.index}")

n_parameters = count_parameters(model)
print(f"Number of trainable parameters: {n_parameters:,}")
#%% OPTIMIZER AND SCHEDULER
BASE_LR = 5e-5 
MIN_LR = 1e-8  
TOTAL_STEPS = sum(len(loader) for loader in train_loaders.values()) * EPOCHS
WARMUP_STEPS = sum(len(loader) for loader in train_loaders.values()) * WARMUP_EPOCHS

optimizer = AdamW(
    model.parameters(),
    lr=BASE_LR,
    betas=(BETA1, BETA2),
    weight_decay=WEIGHT_DECAY
)
def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        # Linear warmup
        return current_step / WARMUP_STEPS
    else:
        # Scaled cosine decay
        scaled_progress = (current_step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * scaled_progress))
        return cosine_decay * (BASE_LR - MIN_LR) / BASE_LR + MIN_LR / BASE_LR
    
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
#%% PRE-TRAIN THE MODEL
pretrained_model = train_lwm(
    model,
    train_loaders,
    val_loaders,
    optimizer,
    scheduler,
    EPOCHS,
    device=device
)