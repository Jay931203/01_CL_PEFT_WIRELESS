from input_preprocess import clone_dataset_scenarios
import numpy as np

dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"
scenario_names = np.array([
    "city_6_miami",
    "city_7_sandiego",
    "city_11_santaclara",
    "city_12_fortworth",
    "city_15_indianapolis",
    "city_18_denver",
    "city_19_oklahoma"
])
#scenario_idxs = np.array([0])
scenario_idxs = np.array([0,1,2,3,4,5,6])
selected_scenario_names = scenario_names[scenario_idxs]
