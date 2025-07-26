from input_preprocess import clone_dataset_scenarios
import numpy as np

dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"
scenario_names = np.array(["city_6_miami"])
scenario_idxs = np.array([0])  # 하나만 선택

selected_scenario_names = scenario_names[scenario_idxs]

clone_dataset_scenarios(
    selected_scenario_names=selected_scenario_names,
    dataset_repo_url=dataset_repo_url,
    model_repo_dir="./"  # 현재 폴더에 scenarios/ 폴더 자동 생성
)
