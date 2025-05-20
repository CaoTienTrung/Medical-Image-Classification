import kagglehub
import shutil
import os

def download_dataset(dataset_name, save_dir):
    path = kagglehub.dataset_download(dataset_name)
    shutil.copytree(path, save_dir, dirs_exist_ok=True)
    
    print("Dataset saved to:", save_dir)

if __name__ == "__main__":
    DATASET_NAME = "mohamedhanyyy/chest-ctscan-images"
    SAVE_DIR = r"F:\Studies\Third_year\Computer_vision\Project\Dataset"
    download_dataset(dataset_name=DATASET_NAME,
                     save_dir=SAVE_DIR)