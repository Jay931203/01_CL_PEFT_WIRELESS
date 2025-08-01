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
from collections import defaultdict


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
train_ratios = [0.8]  # Fraction of data for training [.001, .01, .05, .1, .25, .5, .8]
fine_tuning_status = [None]  # Fine-tuning configurations [None, ["layers.8", "layers.9", "layers.10", "layers.11"], "full"]
test_type = ["backbone", "full"][0]
test_cond = ["Tasks","Snrs"][1]
fine_tune = False
mask=True
snr = 30
epoch_num = 150

continual_learning_type = [None, "LwF","EWC"][1]
#scenario_nums = [    6,    7,    11,    12,    15,    18,    19], 7, 11, 12
scenario_nums = [7]
interation = 7

scenario_list = []
city_order={}

for scenario in scenario_nums:
    all_scenarios = scenarios_list()
    scenario_list.append(all_scenarios[scenario])  # 예: city_6_miami
    city_order[all_scenarios[scenario]] = scenario

new_metrics_by_train_city = {}
#"results/Embedding Regression/1754022362/channel_emb_epoch149_valLoss64.8729_1754022514.pth" #Scenario 7, 15dB Head-baseline(Frozen)
# Avg SNR
#
#"results/Embedding Regression/1753967504/channel_emb_epoch147_valLoss12.7858_1753967562.pth" #Full Scenario, head-baseline
#
#
baseline =  "results/Embedding Regression/1754022362/channel_emb_epoch149_valLoss64.8729_1754022514.pth" #12, head-baseline
best_model_paths = {
    "city_6_miami": None,
    "city_7_sandiego": None,
    "city_11_santaclara": None,
    "city_12_fortworth": None,
    "city_15_indianapolis": None,
    "city_18_denver": None,
    "city_19_oklahoma": None,
}

snr_nums=[]
main_order={}
for i in range(interation):
    #snr_nums = [30, 25, 20, 15, 10, 5, 0]  # ,20,15,10,5,0,-5,-10]
    snr_nums.append(30-(i*5))
    for scenario in scenario_nums:
        all_scenarios = scenarios_list()
        selected_scenario_names = [all_scenarios[scenario]]  # 예: city_6_miami
        new_metrics_by_train_city[selected_scenario_names[0] + f' ({i})'] = []
        base_number = scenario + i * 20
        key = f"{selected_scenario_names[0]} ({i})"
        main_order[key] = base_number

#print(main_order)

for i in range(interation):
    snr = snr_nums[i]
    for scenario in scenario_nums:
        #fine_tune = True
        print(f"Running scenario #{scenario}")

        # 0_3 Scenarios Setting
        all_scenarios = scenarios_list()
        selected_scenario_names = [all_scenarios[scenario]]  # 예: city_6_miami
        print(f"Selected Scenario: {selected_scenario_names[0]}")

        pth_path = best_model_paths.get(selected_scenario_names[0])
        if (pth_path == None):
            fine_tune = True
            pth_path = baseline
            print(f"Path change#{pth_path}")
        # 0_4 Tokenizer(Pre-processing)
        preprocessed_data, labels, raw_chs = tokenizer(
            selected_scenario_names,
            bs_idxs=[3],           # BS index 3번 사용
            load_data=True,       # 이미 있으면 True, 처음 실행이면 False
            task=task,
            n_beams = n_beams,
            mask=mask,
            masking_percent = 0.3,
            snr = snr
        )
        if isinstance(preprocessed_data, dict):
            all_samples = [torch.tensor(sample[0], dtype=torch.float32)
                           for samples in preprocessed_data.values()
                           for sample in samples]
            preprocessed_data = torch.stack(all_samples, dim=0)  # [N, 33, 32]
            print("✅ Model Structure Re-Arranged.")

        pred_tensor = preprocessed_data[:, 1:, :]
        diff = torch.abs(pred_tensor - labels)
        print("(Masking Check) Mean absolute difference:", torch.mean(diff))

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
        # print(subset_data.shape)
        print("✅ Inference Done")
        #sys.exit(1)

        # Step 3: Proposed Fine-tuning
        metrics_by_scenario = {}
        metrics_by_snr = {}
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
                        optimizer_config={"lr": 1e-4},
                        epochs=epoch_num,
                        mask=mask,
                        fine_tune=fine_tune,
                        resume_path = pth_path,
                        device=device,
                        task=task
                    )

                    if (fine_tune):
                        results[fine_tuning_stat_idx][input_type_idx][train_ratio_idx] = Accuracy[-1]
                    else:
                        metrics_by_scenario[selected_scenario_names[0]] = Accuracy[-1]
                    baseline = best_model_path
                    print(f"Path change#{baseline}")

        test_type = ["backbone", "full"][1]

        fine_tune = False

        if (test_cond == 'Snrs'):
            for snr_idx in snr_nums:
                print(f"▶️ Running for scenario: {selected_scenario_names[0]}, {snr_idx}")
                preprocessed_data, labels, raw_chs = tokenizer(
                    selected_scenario_names,
                    bs_idxs=[3],           # BS index 3번 사용
                    load_data=True,       # 이미 있으면 True, 처음 실행이면 False
                    task=task,
                    n_beams = n_beams,
                    mask=mask,
                    masking_percent = 0.3,
                    snr = snr_idx
                )

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
                    epochs=epoch_num,
                    mask=mask,
                    fine_tune=fine_tune,
                    resume_path=baseline,
                    device=device,
                    task=task
                )
                metrics_by_snr[snr_idx] = Accuracy[-1]

        else:
            for scenario in scenario_nums:
                scenario_name = all_scenarios[scenario]  # 예: city_6_miami
                print(f"▶️ Running for scenario: {scenario_name}")
                preprocessed_data, labels, raw_chs = tokenizer(
                    [scenario_name],
                    bs_idxs=[3],           # BS index 3번 사용
                    load_data=True,       # 이미 있으면 True, 처음 실행이면 False
                    task=task,
                    n_beams = n_beams,
                    mask=mask,
                    masking_percent = 0.3,
                    snr = snr
                )

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
                    epochs=epoch_num,
                    mask=mask,
                    fine_tune=fine_tune,
                    resume_path=baseline,
                    device=device,
                    task=task
                )
                metrics_by_scenario[scenario_name] = Accuracy[-1]


        if (test_cond == 'Snrs'):
            metrics = metrics_by_snr
            scenarios = list(metrics_by_snr.keys())
            metric_values = list(metrics_by_snr.values())
        else:
            metrics = metrics_by_scenario
            scenarios = list(metrics_by_scenario.keys())
            metric_values = list(metrics_by_scenario.values())

        from collections import OrderedDict
        import matplotlib.pyplot as plt
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
        plt.savefig(f"loss_plot_{selected_scenario_names[0]}.png")
        #plt.show()
        plt.close()
        for _, val in metrics.items():
            new_metrics_by_train_city[selected_scenario_names[0]+ f' ({i})'].append(val)

import pandas as pd
# 3. DataFrame으로 변환
if (test_cond == 'Snrs'):
    df_nmse = pd.DataFrame(new_metrics_by_train_city, index=snr_nums)
    df_long = df_nmse.reset_index().melt(id_vars='index', var_name='Main City', value_name='NMSE')
    df_long = df_long.rename(columns={'index': 'Test City'})
    df_long['Test City Number'] = pd.factorize(df_long['Test City'])[0]
    df_long['Main City Number'] = pd.factorize(df_long['Main City'])[0]
else:
    df_nmse = pd.DataFrame(new_metrics_by_train_city, index=scenario_list)
    df_long = df_nmse.reset_index().melt(id_vars='index', var_name='Main City', value_name='NMSE')
    df_long = df_long.rename(columns={'index': 'Test City'})
    df_long['Test City Number'] = df_long['Test City'].map(city_order)
    df_long['Main City Number'] = df_long['Main City'].map(main_order)

df_long_sorted = df_long.sort_values(by=['Test City Number', 'Main City Number'])
cl_fixed = df_long_sorted[df_long_sorted['Main City Number'] >= df_long_sorted['Test City Number']]

# 7. 그래프 그리기
plt.figure(figsize=(10, 6))
for test_city in df_long_sorted['Test City'].unique():
    subset = df_long_sorted[df_long_sorted['Test City'] == test_city]
    #label = f"{city_order[test_city]}_{test_city.split('_')[2]}"
    label = f"{test_city}"
    print(f"{test_city}: {len(subset)} rows")
    plt.plot(
        subset['Main City Number'],
        subset['NMSE'],
        marker='o',
        label=label
    )

# sorted_items = sorted(main_order.items(), key=lambda x: x[1])  # value 기준 정렬
# xtick_positions = [num for city, num in sorted_items]
# xtick_labels = [city for city, num in sorted_items]
#zplt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=45, ha='right')
# plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=45, ha='right')
# for i in range(interation):
#     plt.axvline(x=20*(i+1), color='gray', linestyle='--', linewidth=1)
plt.xticks(ticks=range(len(snr_nums)), labels=snr_nums, rotation=45, ha='right')
plt.xlabel("Trained On")
plt.ylabel("NMSE")
plt.title("Sequential Testing")
plt.legend(title="Evaluated On", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"final_plot(original)_snr.png")
plt.close()

plt.figure(figsize=(10, 6))
for test_city in cl_fixed['Test City'].unique():
    subset = cl_fixed[cl_fixed['Test City'] == test_city]
    label = f"{test_city}"
    plt.plot(
        subset['Main City Number'],
        subset['NMSE'],
        marker='o',
        label=label
    )

plt.xticks(ticks=range(len(snr_nums)), labels=snr_nums, rotation=45, ha='right')
# for i in range(interation):
#     plt.axvline(x=20*(i+1), color='gray', linestyle='--', linewidth=1)
plt.xlabel("Main City Number (Trained On)")
plt.ylabel("NMSE")
plt.title("NMSE vs Trained City (Sequential Testing per Test City)")
plt.legend(title="Test City (Evaluated On)", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"final_plot(truncated)_snr.png")
plt.close()