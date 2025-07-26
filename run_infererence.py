import torch
import torch.nn as nn
import warnings
import numpy as np

from input_preprocess import tokenizer, scenarios_list
from inference import lwm_inference
from utils import prepare_loaders
from lwm_model import lwm

def compute_nmse(pred, target):
    """
    pred, target: [B, S, D] tensor
    """
    mse = torch.sum((pred - target) ** 2, dim=(-1, -2))  # sum over S and D
    power = torch.sum(target ** 2, dim=(-1, -2))          # sum over S and D
    nmse = torch.mean(mse / (power + 1e-10))              # avoid division by zero
    return nmse.item()

warnings.filterwarnings("ignore", category=UserWarning)

# --------- Step 2: Task 및 설정 ---------
n_beams = 16  # Beam Prediction일 경우 사용
task = 'LoS/NLoS Classification'  # 또는 'Beam Prediction'
task_type = 'classification'      # 또는 'regression'
visualization_method = 'tsne'     # 'pca', 'umap'도 가능
input_type = 'cls_emb'            # 'channel_emb', 'raw' 도 가능
train_ratio = 0.1                 # 10% 데이터만 학습에 사용
fine_tune_layers = None           # fine-tuning 하지 않음

# --------- Step 3: 시나리오 선택 ---------
all_scenarios = scenarios_list()
selected_scenario_names = [all_scenarios[6]]  # 예: city_6_miami

print(f"Selected Scenario: {selected_scenario_names[0]}")

# --------- Step 4: 데이터 전처리 (Tokenizer) ---------
preprocessed_data, labels, raw_chs = tokenizer(
    selected_scenario_names,
    bs_idxs=[3],           # BS index 3번 사용
    load_data=True,       # 이미 있으면 True, 처음 실행이면 False
    task=task,
    n_beams=n_beams,
    masking_percent = 0.00,
)

print("✅ Tokenization 완료")
print("preprocessed_data shape:", preprocessed_data.shape)
print("labels shape:", labels.shape)

# --------- Step 5: LWM 모델 로드 ---------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = lwm().to(device)

# pretrained model 로드
model_name = "model.pth"
state_dict_path = f"models/{model_name}"

state_dict = torch.load(state_dict_path, map_location=device)
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.eval()

print("✅ Model loaded successfully.")

# --------- Step 6: Inference ---------

# 전처리된 sample 중 일부만 사용
samples = torch.tensor(preprocessed_data[:1000], dtype=torch.float32).to(device)
true_labels = torch.tensor(labels[:1000], dtype=torch.long).to(device)

# 모델 추론 (CLS token만 추출)
with torch.no_grad():
    sequence_output, _ = model(samples)     # attention map은 무시
    cls_rep = sequence_output[:, 0, :]      # [CLS] token 추출
    sample_noCLS = torch.tensor(preprocessed_data[:1000, 1:, :], dtype=torch.float32).to(device)  # shape: [1000, 32, 32]
    predicted = model.decoder(sequence_output[:, 1:, :])  # shape: [1000, 32, 32]


print("✅ Inference 완료")
print("Representation shape:", cls_rep.shape)
print("Decpder shape:", predicted.shape)

# --------- Step 7: Task결과 시각화 or NMSE ---------
#from utils import visualize_embeddings
#visualize_embeddings(cls_rep.cpu(), true_labels.cpu(), method="tsne", label="LWM Representation")
nmse = compute_nmse(predicted, sample_noCLS)
print(f"🔎 NMSE between input and embedding output (no CLS): {nmse:.4f}")

# --------- Step 8: Task결과 시각화 or NMSE ---------

