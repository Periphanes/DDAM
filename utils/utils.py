import torch
import numpy as np

# Original get_datasets_statistics function in octo/octo/utils/data_utils.py
# TODO : implement caching for faster future use (in original function)
def get_dataset_statistics(leDataset):
    actions = torch.stack(leDataset.hf_dataset["action"]).numpy()
    proprios = torch.stack(leDataset.hf_dataset["observation.state"]).numpy()

    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "p99": np.quantile(actions, 0.99, 0).tolist(),
            "p01": np.quantile(actions, 0.01, 0).tolist(),
        },
        "num_transitions" : leDataset.meta.total_frames,
        "num_trajectories" : leDataset.meta.total_episodes,
    }
    metadata["proprios"] = {
        "mean": proprios.mean(0).tolist(),
        "std": proprios.std(0).tolist(),
        "max": proprios.max(0).tolist(),
        "min": proprios.min(0).tolist(),
        "p99": np.quantile(proprios, 0.99, 0).tolist(),
        "p01": np.quantile(proprios, 0.01, 0).tolist(),
    }

    return metadata