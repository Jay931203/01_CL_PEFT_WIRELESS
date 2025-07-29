from input_preprocess import clone_dataset_scenarios
import subprocess
import os
import shutil

def clone_dataset_scenario(repo_url, model_repo_dir="./LWM", scenarios_dir="scenarios"):
    """
    Clones all scenarios from a repository, ensuring all files (small and large) are downloaded.

    Args:
        repo_url (str): URL of the Git repository
        model_repo_dir (str): Path to the model repository
        scenarios_dir (str): Directory name for storing scenarios
    """
    # Ensure we're in the correct directory structure
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "LWM":
        model_repo_dir = "."

    # Create the scenarios directory if it doesn't exist
    scenarios_path = os.path.join(model_repo_dir, scenarios_dir)
    os.makedirs(scenarios_path, exist_ok=True)

    # Store the original working directory
    original_dir = os.getcwd()

    try:
        # Clean up any existing temp directory
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)

        # Clone the entire repository (including all files)
        print(f"Cloning entire repository into temporary directory...")
        subprocess.run([
            "git", "clone",
            repo_url,
            scenarios_path
        ], check=True)

        # Navigate to the temporary clone directory
        os.chdir(scenarios_path)

        # Pull all files using Git LFS
        print(f"Pulling all files using Git LFS...")
        subprocess.run(["git", "lfs", "install"], check=True)  # Ensure LFS is installed
        subprocess.run(["git", "lfs", "pull"], check=True)  # Pull all LFS files

        print(f"Successfully cloned all scenarios into {scenarios_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error cloning scenarios: {str(e)}")
    finally:
        # Clean up temporary directory
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)
        # Return to original directory
        os.chdir(original_dir)
# import numpy as np
#
# dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"
# scenario_names = np.array([
#     "city_6_miami",
#     "city_7_sandiego",
#     "city_11_santaclara",
#     "city_12_fortworth",
#     "city_15_indianapolis",
#     "city_18_denver",
#     "city_19_oklahoma"
# ])
# #scenario_idxs = np.array([0])
# scenario_idxs = np.array([0,1,2,3,4,5,6])
# selected_scenario_names = scenario_names[scenario_idxs]

# Step 1: Clone the model repository (if not already cloned)
model_repo_url = "https://huggingface.co/wi-lab/lwm"
model_repo_dir = "./"

# if not os.path.exists(model_repo_dir):
#     print(f"Cloning model repository from {model_repo_url}...")
#     subprocess.run(["git", "clone", model_repo_url, model_repo_dir], check=True)



import numpy as np
dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"  # Base URL for dataset repo

# Clone the requested scenarios
clone_dataset_scenario(dataset_repo_url, model_repo_dir)