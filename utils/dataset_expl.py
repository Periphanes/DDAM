import torch
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from huggingface_hub import HfApi
from pprint import pprint

repo_id = "lerobot/aloha_sim_transfer_cube_human_image"
dataset_root = "/mnt/hdd/john/lerobot_dataset/aloha_sim_transfer_cube_human_image"


ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
le_dataset = LeRobotDataset(repo_id, root=dataset_root)


for x in le_dataset.hf_dataset["action"]:
    if x.shape[0] != 14:
        print(x.shape)

print(le_dataset.hf_dataset["action"][0].shape)

actions = torch.stack(le_dataset.hf_dataset["action"]).numpy()
print(actions.shape)
print(actions.dtype)

proprios = torch.stack(le_dataset.hf_dataset["observation.state"]).numpy()
print(proprios.shape)

print(actions.mean(0).tolist())