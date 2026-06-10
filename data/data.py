import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Downloaded to:", path)

# Copy files to current working directory
current_dir = os.getcwd()

for file_name in os.listdir(path):
    src = os.path.join(path, file_name)
    dst = os.path.join(current_dir, file_name)

    if os.path.isfile(src):
        shutil.copy(src, dst)

print("Files copied to:", current_dir)