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
train_ratios = [.0]  # Fraction of data for training [.001, .01, .05, .1, .25, .5, .8]
fine_tuning_status = [None]  # Fine-tuning configurations [None, ["layers.8", "layers.9", "layers.10", "layers.11"], "full"]
test_type = ["backbone", "full"][0]
fine_tune = False
mask=True

scenario_nums = [    6,    7,    11,    12,    15,    18,    19]

best_model_paths = {
    "city_6_miami": "results/Embedding Regression/1753710883/channel_emb_epoch199_valLoss27.9700_1753711888.pth",
    "city_7_sandiego": "results/Embedding Regression/1753710883/channel_emb_epoch199_valLoss27.9700_1753711888.pth",
    "city_11_santaclara": None,
    "city_12_fortworth": None,
    "city_15_indianapolis": None,
    "city_18_denver": None,
    "city_19_oklahoma": None,
}

# 0_3 Scenarios Setting
all_scenarios = scenarios_list()
selected_scenario_names = [all_scenarios[7]]  # 예: city_6_miami
print(f"Selected Scenario: {selected_scenario_names[0]}")

pth_path = best_model_paths.get(selected_scenario_names[0])

# 0_4 Tokenizer(Pre-processing)
preprocessed_data, labels, raw_chs = tokenizer(
    selected_scenario_names,
    bs_idxs=[3],           # BS index 3번 사용
    load_data=True,       # 이미 있으면 True, 처음 실행이면 False
    task=task,
    n_beams = n_beams,
    mask=mask,
    masking_percent = 0.3
)
if isinstance(preprocessed_data, dict):
    all_samples = [torch.tensor(sample[0], dtype=torch.float32)
                   for samples in preprocessed_data.values()
                   for sample in samples]
    preprocessed_data = torch.stack(all_samples, dim=0)  # [N, 33, 32]
    print("✅ Model Structure Re-Arranged.")


print(f"ℹ️ preprocessed_data shape: {preprocessed_data.shape}")

# Step 1: LWM Load
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = lwm().to(device)
model_name = "model.pth"
state_dict_path = f"models/{model_name}"
state_dict = torch.load(state_dict_path, map_location=device)
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
print("✅ Model loaded successfully.")

subset_data = preprocessed_data[:]
subset_labels = labels[:]
# #Step 2: Inference ---------
print(f"💻 Using device: {device}")
print(f"🧠 GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
chs = lwm_inference(
    model,
    subset_data,
    input_type= input_types[0],
    device=device,
    task=task,
    task_type=task_type,
    batch_size=128,
    visualization=False,
    test_type = test_type,
    mask=mask,
    labels=subset_labels,
    visualization_method=visualization_method
)
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
                batch_size=128,
                seed=42
            )
            # Fine-tune LWM
            fine_tuned_model, best_model_path, train_losses, val_losses, Accuracy, attn_maps_ft, best_model_path = finetune(
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
                epochs=50,
                mask=mask,
                fine_tune=fine_tune,
                resume_path = pth_path, #0.037
                device=device,
                task=task
            )
            if (fine_tune):
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

metrics_by_scenario = {}
chs, metric = lwm_inference(
    fine_tuned_model,
    val_samples,
    input_type = input_types[0],
    device=device,
    batch_size=128,
    task=task,
    task_type=task_type,
    test_type=test_type,
    visualization=False,
    mask=True,
    labels=val_targets,
    resume_path = best_model_path,
    visualization_method=visualization_method
)
print(val_samples.shape)
print(val_targets.shape)
metrics_by_scenario[selected_scenario_names[0]] = metric

for scenario in scenario_nums:
    scenario_name = all_scenarios[scenario]  # 예: city_6_miami
    if scenario_name == selected_scenario_names[0]:
        continue
    print(f"▶️ Running for scenario: {scenario_name}")
    preprocessed_data, labels, raw_chs = tokenizer(
        [scenario_name],
        bs_idxs=[3],           # BS index 3번 사용
        load_data=True,       # 이미 있으면 True, 처음 실행이면 False
        task=task,
        n_beams = n_beams,
        mask=mask,
        masking_percent = 0.3
    )
    if isinstance(preprocessed_data, dict):
        all_samples = [torch.tensor(sample[0], dtype=torch.float32)
                       for samples in preprocessed_data.values()
                       for sample in samples]
        preprocessed_data = torch.stack(all_samples, dim=0)  # [N, 33, 32]
        print("✅ Model Structure Re-Arranged.")
        print(preprocessed_data.shape)
        labels = labels.contiguous().view(labels.size(0), -1)
        print(labels.shape)

    chs, metric = lwm_inference(
        fine_tuned_model,
        preprocessed_data,
        input_type=input_types[0],
        device=device,
        batch_size=128,
        task=task,
        task_type=task_type,
        test_type=test_type,
        visualization=False,
        mask=True,
        labels=labels,
        resume_path=best_model_path,
        visualization_method=visualization_method
    )
    metrics_by_scenario[scenario_name] = metric


import matplotlib.pyplot as plt
scenarios = list(metrics_by_scenario.keys())
metric_values = list(metrics_by_scenario.values())

# Bar Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(scenarios, metric_values)
plt.xticks(rotation=45, ha='right')
plt.ylabel('NMSE' if task_type == 'regression' else 'F1 Score')
plt.title(f"📊 Performance by Scenario (Main: {selected_scenario_names[0]})")

# 값 위에 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()