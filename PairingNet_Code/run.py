import os
import subprocess

path = os.path.dirname(os.path.abspath(__file__))
run_file = os.path.join(path, "PairingNet_train_val_test.py")

subprocess.run(["python", run_file, "--model_type=matching_train"])
subprocess.run(["python", run_file, "--model_type=matching_test"])
subprocess.run(["python", run_file, "--model_type=save_stage1_feature"])

cmd = f"torchrun --nproc_per_node=4 {run_file} --model_type=searching_train"
subprocess.run(cmd, shell=True, cwd=path)

# cmd = f"python texture_countour_double_GCN.py --model_type=searching_test"
# subprocess.run(cmd, shell=True, cwd=path)