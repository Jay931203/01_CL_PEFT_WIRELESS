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
n_beams = 4  # Beam Prediction일 경우 사용
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

# Step 1: LWM Load
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = lwm().to(device)
model_name = "model.pth"
state_dict_path = f"models/{model_name}"
state_dict = torch.load(state_dict_path, map_location=device)
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
print("✅ Model loaded successfully.")

subset_data = preprocessed_data[:1000]
subset_labels = labels[:1000]
# #Step 2: Inference ---------
chs = lwm_inference(
    model,
    subset_data,
    input_type= input_types[0],
    device=device,
    task=task,
    task_type=task_type,
    batch_size=128,
    visualization=True,
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
                epochs=3,
                device=device,
                task=task
            )
            results[fine_tuning_stat_idx][input_type_idx][train_ratio_idx] = Accuracy[-1]

# Step 3_1: Proposed Fine-Tuning Results
markers = ['o', 's', 'D']
labels = ['CHS Emb', 'Raw']
fine_tuning_status_labels = []
line_styles = []
if None in fine_tuning_status:
    fine_tuning_status_labels.append('No FT')
    line_styles.append('-')
if "partial" in fine_tuning_status:
    fine_tuning_status_labels.append('Partial FT')
    line_styles.append('--')
if "full" in fine_tuning_status:
    fine_tuning_status_labels.append('Full FT')
    line_styles.append(':')
colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
plt.figure(figsize=(12, 8), dpi=500)
for ft_idx, (ft_status_label, line_style) in enumerate(zip(fine_tuning_status_labels, line_styles)):
    for idx, (marker, label, color) in enumerate(zip(markers, labels, colors)):
        if label == "Raw" and ft_status_label != "No FT":
            continue
        plt.plot(
            train_ratios,
            results[ft_idx, idx],
            marker=marker,
            linestyle=line_style,
            label=f"{label} ({ft_status_label})",
            color=color,
            linewidth=3,
            markersize=9
        )
plt.xscale('log')
plt.xlabel("Train Ratio", fontsize=20)
plt.ylabel("Acc(F1-Score or NMSE)", fontsize=20)
plt.legend(fontsize=17, loc="best")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.show()


test_type = ["backbone", "full"][1]

# 3_2:Comparing the Raw Space with Fine-Tuned Embedding Space
chs = lwm_inference(
    fine_tuned_model.model,
    subset_data,
    input_type="cls_emb",
    device=device,
    batch_size=64,
    task=task,
    task_type=task_type,
    test_type=test_type[1],
    visualization=True,
    labels=subset_labels,
    visualization_method=visualization_method
)

#
# # --------- Step 8: Reconstruction Task: Fine-tuning Head 정의 ---------
# # ----- Decoder 추출 -----
# decoded_path = 'decoded_tensor.pt'
# target_path = 'pseudo_target_from_mask0.pt'
#
# # ----- 전체 데이터 사용 -----
# samples = torch.tensor(preprocessed_data, dtype=torch.float32).to(device)
# labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
#
# if (mask == False):
#     pseudo_target = preprocessed_data[:, 1:, :].clone()
#     torch.save(pseudo_target, target_path)
#     print(f"✅ pseudo_target shape: {pseudo_target.shape}")  # [N, 32, 32] 예상
#     print(f"Mean: {pseudo_target.mean():.4f}, Std: {pseudo_target.std():.4f}, Max: {pseudo_target.max():.4f}, Min: {pseudo_target.min():.4f}")
#     print(f"✅ (NO-MASK) pseudo_target_saved.")
#     print(preprocessed_data.shape)  # torch.Size([11688, 33, 32])
#     print(preprocessed_data[0, 1, :10])  # 두 번째 row (pseudo_target 시작 부분)
#
# batch_size = 128  # 원하는 배치 크기 설정
#
# if (0):
# #if not os.path.exists(decoded_path) or not os.path.exists(target_path):
#     print("🔥 No saved decoder output. Running decoder batched...")
#     start_time = time.time()
#     dataset = TensorDataset(samples)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     decoded_list = []
#     target_list = []
#
#     model.eval()
#     with torch.no_grad():
#         for xb in loader:  # xb: [B, 33, 32], yb: [B, 32, 32]
#             xb = xb[0].to(device)  # [B, 33, 32]
#             reps, _ = model(xb)    # [B, 33, D]
#             decoded_batch = model.decoder(reps[:, 1:, :]) + model.decoder_bias  # [B, 32, 32]
#             decoded_list.append(decoded_batch.cpu())
#             #target_list.append(xb[:, 1:, :].cpu())  # target도 함께 저장
#             #print(f"⏱️ Each decoding time: {time.time() - start_time:.2f} sec")
#
#     decoded = torch.cat(decoded_list, dim=0)
#     target = torch.load(target_path)
#     torch.save(decoded, decoded_path)
#
#     print(f"✅ Decoding completed and saved.")
#     print(f"📐 decoded shape: {decoded.shape}, target shape: {target.shape}")
#     print(f"⏱️ Total decoding time: {time.time() - start_time:.2f} sec")
# else:
#     # 재사용 시 로드
#     decoded = torch.load(decoded_path).to(device)
#     target = torch.load(target_path).to(device)
#     print("✅ decoded & target loaded")
#
#
# # --------- RescaleHead 정의 ---------
# class RescaleHead(nn.Module):
#     def __init__(self, base_head, channels=32):
#         super().__init__()
#         self.base = base_head
#         self.scale = nn.Parameter(torch.ones(1, channels, 1))  # [1, 32, 1] ← column별 scale
#         self.bias = nn.Parameter(torch.zeros(1, channels, 1))
#
#     def forward(self, x):
#         return self.base(x) * self.scale + self.bias
#
# # Head 정의
# reconstruct_head = nn.Sequential(
#     #nn.LayerNorm([32, 32]),  # ← decoder 출력(32×32) 자체를 정규화
#     #nn.Flatten(),
#     nn.Linear(32, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32)
# ).to(device)
#
# reconstruct_head = RescaleHead(reconstruct_head).to(device)
#
# # Freeze 모델
# for param in model.parameters():
#     param.requires_grad = False
#
# # ----- Train/Val 분할 -----
# decoded = torch.load(decoded_path)  # shape: [N, 32, 32]
# target = torch.load(target_path).to(torch.float32)
#
# decoded_train, decoded_val, target_train, target_val = train_test_split(
#     decoded, target, test_size=0.2, random_state=42
# )
# decoded_train = decoded_train.to(device)
# target_train = target_train.to(device)
# decoded_val = decoded_val.to(device)
# target_val = target_val.to(device)
#
# # ----- Head Parameter Load -----
# head_path = 'reconstruct_head.pth'
# if (0): #os.path.exists(head_path):
#     reconstruct_head.load_state_dict(torch.load(head_path, map_location=device))
#     reconstruct_head.eval()
#     print("✅ reconstruct_head loaded")
# else:
#     # Optimizer
#     optimizer = torch.optim.Adam(reconstruct_head.parameters(), lr=1e-3)
#     loss_fn = torch.nn.MSELoss()
#     batch_size = 128
#
#     # DataLoader
#     train_dataset = TensorDataset(decoded_train, target_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
#     # 기록 리스트
#     loss_list, nmse_list, epochs = [], [], []
#
#     # ---- Fine-tuning (multi-epoch) ----
#     reconstruct_head.train()
#     best_nmse = float('inf')
#     patience = 5
#     patience_counter = 0
#
#     for epoch in range(50):
#         epoch_start = time.time()
#         running_loss = 0.0
#
#         for xb, yb in train_loader:
#             xb = xb.to(device)
#             yb = yb.to(device)
#
#             optimizer.zero_grad()
#             prediction = reconstruct_head(xb)
#             loss = loss_fn(prediction, yb)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item() * xb.size(0)
#
#         avg_loss = running_loss / len(train_dataset)
#         epoch_time = time.time() - epoch_start
#         print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, epoch time: {epoch_time:.4f}")
#         loss_list.append(avg_loss)
#
#         # ---- Validation 평가 및 Early Stopping ----
#         if (epoch + 1) % 10 == 0:
#             with torch.no_grad():
#                 # ⚠️ 딱 한 번만 랜덤 서브셋 뽑기
#                 subset_ratio = 0.1
#                 N_val = decoded_val.shape[0]
#                 subset_size = int(N_val * subset_ratio)
#                 indices = torch.randperm(N_val)[:subset_size]
#                 decoded_val_subset = decoded_val[indices]
#                 target_val_subset = target_val[indices]
#
#                 reconstruct_head.eval()
#                 val_preds = reconstruct_head(decoded_val_subset)
#                 val_nmse = compute_nmse(val_preds, target_val_subset)
#                 reconstruct_head.train()
#                 print(f"[Epoch {epoch + 1}] 🧪 Validation NMSE: {val_nmse:.4f}")
#
#                 if val_nmse < best_nmse:
#                     best_nmse = val_nmse
#                     patience_counter = 0
#                     torch.save(reconstruct_head.state_dict(), "best_head.pth")
#                     print(f"✅ Best head saved (Val NMSE: {best_nmse:.4f})")
#                 else:
#                     patience_counter += 1
#                     print(f"⏳ Patience: {patience_counter}/{patience}")
#                 if patience_counter >= patience:
#                     print("🛑 Early stopping triggered!")
#                     break
#
#         # ---- 10 epoch마다 NMSE 측정 ----
#         if (epoch + 1) % 10 == 0:
#             with torch.no_grad():
#                 preds = reconstruct_head(decoded).detach()
#                 nmse = compute_nmse(preds, target)
#                 nmse_list.append(nmse)
#                 epochs.append(epoch + 1)
#                 print(f"[Epoch {epoch + 1}] 🔎 NMSE: {nmse:.4f}")
#
#     torch.save(reconstruct_head.state_dict(), head_path)
#     print("✅ Head parameters saved to reconstruct_head.pth")
#
#     # ---- 시각화 ----
#     fig, ax1 = plt.subplots(figsize=(8, 5))
#     # 🎯 Loss (왼쪽 y축)
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Loss", color="tab:blue")
#     ax1.plot(range(1, len(loss_list)+1), loss_list, color="tab:blue", label="Loss")
#     ax1.tick_params(axis="y", labelcolor="tab:blue")
#     # 🎯 NMSE (오른쪽 y축)
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("NMSE", color="tab:red")
#     ax2.plot(epochs, nmse_list, marker='o', linestyle='--', color="tab:red", label="NMSE")
#     ax2.tick_params(axis="y", labelcolor="tab:red")
#     # 🎯 마무리
#     plt.title("Loss and NMSE during Fine-Tuning")
#     fig.tight_layout()
#     plt.grid(True)
#     plt.savefig("loss_nmse_plot.png")
#     plt.show()
#
# # → 이 경우 decoded.shape = target.shape = [전체 샘플 수, 32, 32]
# with torch.no_grad():
#     preds = reconstruct_head(decoded_val)
#     nmse = compute_nmse(preds, target_val)
#     print(f"🔎 NMSE for Reconstruction after head (Full Eval): {nmse:.4f}")
