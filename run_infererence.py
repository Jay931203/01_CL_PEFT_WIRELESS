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
input_types = ["channel_emb"]  # Supported input types: + cls_emb + "raw"
train_ratios = [.8]  # Fraction of data for training [.001, .01, .05, .1, .25, .5, .8]
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
    masking_percent = 0.30
)
if isinstance(preprocessed_data, dict):
    all_samples = [torch.tensor(sample[0], dtype=torch.float32)
                   for samples in preprocessed_data.values()
                   for sample in samples]
    preprocessed_data = torch.stack(all_samples, dim=0)  # [N, 33, 32]
    print("✅ Model Structure Re-Arranged.")

# Step 1: LWM Load
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = lwm().to(device)
model_name = "model.pth"
state_dict_path = f"models/{model_name}"
state_dict = torch.load(state_dict_path, map_location=device)
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
print("✅ Model loaded successfully.")

subset_data = preprocessed_data[:0]
subset_labels = labels[:0]
# #Step 2: Inference ---------
print(f"💻 Using device: {device}")
print(f"🧠 GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
# chs = lwm_inference(
#     model,
#     subset_data,
#     input_type= input_types[0],
#     device=device,
#     task=task,
#     task_type=task_type,
#     batch_size=128,
#     visualization=False,
#     test_type = test_type,
#     mask=mask,
#     labels=subset_labels,
#     visualization_method=visualization_method
# )
print("✅ Inference Done")

# if task == 'Embedding Regression':
#     print("✅ No Fine-tuning. !!!!!TEST-DONE!!!!!")
#     sys.exit(1)

# Step 3: Proposed Fine-tuning
results = np.zeros((len(fine_tuning_status), len(input_types), len(train_ratios)))




for fine_tuning_stat_idx, fine_tuning_stat in enumerate(fine_tuning_status):
    for input_type_idx, input_type in enumerate(input_types):
        if input_type == "raw" and fine_tuning_stat is not None:
            continue
        selected_patches_idxs = None
        for train_ratio_idx, train_ratio in enumerate(train_ratios):
            print(f"\nfine-tuning status: {fine_tuning_stat}")
            print(f"input type: {input_type}")
            print(f"train ratio: {train_ratio}\n")

            # Prepare data loaders
            train_loader, val_loader, samples, target = prepare_loaders(
                preprocessed_data=preprocessed_data,
                labels=labels,
                selected_patches_idxs=selected_patches_idxs,
                input_type=input_type,
                task_type=task_type,
                train_ratio=train_ratio,
                batch_size=64,
                seed=42
            )

            # Fine-tune LWM
            fine_tuned_model, best_model_path, train_losses, val_losses, Accuracy, attn_maps_ft = finetune(
                base_model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                task_type=task_type,
                input_type=input_type,
                num_classes=n_beams if task == 'Beam Prediction' else 2 if task == 'LoS/NLoS Classification' else None,
                output_dim=target.shape[-1] if task_type == 'regression' else None,
                use_custom_head=True,
                fine_tune_layers=fine_tuning_stat,
                optimizer_config={"lr": 1e-3},
                epochs=20,
                mask=mask,
                device=device,
                task=task
            )
            results[fine_tuning_stat_idx][input_type_idx][train_ratio_idx] = Accuracy[-1]

test_type = ["backbone", "full"][1]

# 3_2:Comparing the Raw Space with Fine-Tuned Embedding Space
val_subset = val_loader.dataset
full_dataset = val_subset.dataset
val_indices = val_subset.indices
if input_types[0] == "raw":
    val_samples = full_dataset.tensors[0][val_indices]  # shape: [B, 992]
elif input_types[0] == "channel_emb":
    val_samples = preprocessed_data[val_indices]
val_targets = full_dataset.tensors[1][val_indices]

chs = lwm_inference(
    fine_tuned_model,
    val_samples,
    input_type = input_types[0],
    device=device,
    batch_size=64,
    task=task,
    task_type=task_type,
    test_type=test_type,
    visualization=False,
    mask=True,
    labels=val_targets,
    visualization_method=visualization_method
)