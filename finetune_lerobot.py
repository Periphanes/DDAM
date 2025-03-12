"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...
"""
from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead, DiscreteDiffusionActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

import torch
import lerobot

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from huggingface_hub import HfApi
from pprint import pprint
import torch.nn.functional as F
from torchvision.transforms import Resize

from einops import rearrange
import numpy as np
from transformers import AutoProcessor
from torch.utils.data._utils.collate import default_collate

from utils.utils import get_dataset_statistics

MASK_TOKEN = 2048
PAD_TOKEN = 2049

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 5, "Batch size for finetuning.")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)

flags.DEFINE_integer("action_horizon", 75, "action horizon sampled before tokenization")
flags.DEFINE_integer("token_horizon", 30, "number of tokens used as action chunk horizon")

def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    # wandb.init(name="finetune_aloha", project="octo")

    repo_id = "lerobot/aloha_sim_transfer_cube_human_image"
    dataset_root = "/mnt/hdd/john/lerobot_dataset/aloha_sim_transfer_cube_human_image"

    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)

    # Resize image to 256 x 256 for faster inference
    image_transforms = Resize((256, 256))

    # Take 50 as action chunk size (1s in 50Hz ALOHA Sim setup)
    delta_timestamps = {
        "observation.images.top": [0],
        "observation.state": [0],
        "action" : [t / 50 for t in range(FLAGS.action_horizon)]
    }

    le_dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps, image_transforms=image_transforms)
    print(f"Number of episodes selected: {le_dataset.num_episodes}")
    print(f"Number of frames selected: {le_dataset.num_frames}")
    le_dataset_stats = get_dataset_statistics(le_dataset)

    camera_key = le_dataset.meta.camera_keys[0]

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    text_processor = pretrained_model.text_processor
    tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    def collate_lerobot(orig_batch):
        new_batch = {}

        orig_batch = default_collate(orig_batch)

        # LeRobot Batch (Orig Batch)
        # action : [32, 50, 14] torch tensor (batch #, action_horizon, proprio control joint #)
        # action_is_pad : [32, 50] torch boolean (batch #, action_horizon)
            # -> True if padding

        # RLDS Batch (New Batch)
        # action : (32, 1, 50, 14) numpy array -> second dim = window size
        # action_pad_mask : (32, 1, 50, 14) numpy array
            # -> True if NOT padding, need to invert from LeRobot Padding

        action_org = orig_batch["action"]
        action_pad_org = orig_batch["action_is_pad"]

        tokenized_action_batch = []
        tokenized_action_pad_batch = []

        # tokenize actions in batch and match length to FLAGS.token_horizon, adding PAD_TOKEN if needed
        for batch_num_i in range(FLAGS.batch_size):
            valid_actions = action_org[batch_num_i][~action_pad_org[batch_num_i]]
            tokens = tokenizer(valid_actions)[0]
            if len(tokens) < FLAGS.token_horizon:
                tokens = tokens + [PAD_TOKEN] * (FLAGS.token_horizon - len(tokens))
                pad = [True] * (len(tokens)) + [False] * (FLAGS.token_horizon - len(tokens))
            else:
                tokens = tokens[:FLAGS.token_horizon]
                pad = [True] * len(tokens)
            tokenized_action_batch.append(tokens)
            tokenized_action_pad_batch.append(pad)
        tokenized_action_batch = np.array(tokenized_action_batch) # (32,30) -> (batch_size, token_horizon)
        tokenized_action_pad_batch = np.array(tokenized_action_pad_batch) # (32, 30) -> (batch_size, token_horizon)

        new_batch["action"] = tokenized_action_batch
        new_batch["action_pad_mask"] = tokenized_action_pad_batch

        # Orig Batch
        # observation.images.top : (32, 1, 3, 256, 256) tensor
        # observation.state : (32, 1, 14) tensor
        # timestamp : (32) tensor
        # next.done : (32) tensor
        # task : list

        # New Batch
        # observation.image_primary : (32, 1, 256, 256, 3) numpy array
        # observation.proprio : (32, 1, 14) numpy array
        # observation.timestep : (32, 1) numpy array
        # observation.pad_mask_dict : numpy array
        # observation.timestep_pad_mask : (32, 1) numpy array
        # observation.task_complete : (32, 1, 50) numpy array

        new_batch["observation"] = {}
        new_batch["observation"]["image_primary"] = rearrange(orig_batch["observation.images.top"], "b n c h w -> b n h w c").numpy()
        new_batch["observation"]["proprio"] = orig_batch["observation.state"].numpy()

        new_batch["observation"]["task_complete"] = tokenized_action_pad_batch[:, None, :] # (32, 1, 30)

        new_batch["task"] = {}
        new_batch["task"]["language_instruction"] = text_processor.encode(
            orig_batch["task"]
        )
        new_batch["task"]["pad_mask_dict"] = {}
        new_batch["task"]["pad_mask_dict"]["language_instruction"] = np.array([True] * FLAGS.batch_size)

        new_batch["timestamp"] = (orig_batch["timestamp"] * 50).to(torch.int32).view(-1, 1).numpy()
        # new_batch["timestamp"] = torch.IntTensor(orig_batch["timestamp"] * 50).numpy()

        new_batch["observation"]["timestep_pad_mask"] = np.array([[True] for _ in range(FLAGS.batch_size)])

        new_batch["observation"]["pad_mask_dict"] = {}
        new_batch["observation"]["pad_mask_dict"]["image_primary"] = np.array([[True] for _ in range(FLAGS.batch_size)])
        new_batch["observation"]["pad_mask_dict"]["proprio"] = np.array([[True] for _ in range(FLAGS.batch_size)])
        new_batch["observation"]["pad_mask_dict"]["timestep"] = np.array([[True] for _ in range(FLAGS.batch_size)])


        # print(new_batch["observation"]["image_primary"].shape)
        # print(new_batch["observation"]["proprio"].shape)
        # print(new_batch["observation"]["task_complete"].shape)
        # print(new_batch["observation"]["timestep_pad_mask"].shape)

        # exit(0)

        return new_batch

    le_dataloader = iter(torch.utils.data.DataLoader(
        le_dataset,
        batch_size=FLAGS.batch_size,
        drop_last=True,
        collate_fn=collate_lerobot
    ))

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    # logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=50,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )

    # train_data_iter = (
    #     dataset.repeat()
    #     .unbatch()
    #     .batch(FLAGS.batch_size)
    #     .iterator()
    # )

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)
    ex_le_batch = next(le_dataloader)

    # print(example_batch["observation"]["image_primary"].shape)
    # print(example_batch["observation"]["proprio"].shape)
    # print(example_batch["observation"]["timestep"])
    # print(example_batch["observation"]["pad_mask_dict"])
    # print(example_batch["observation"]["timestep_pad_mask"])
    # print(example_batch["observation"]["task_completed"][0], "\n")
    # print(example_batch["task"])

    # print(ex_le_batch["observation.images.top"].shape)
    # print(ex_le_batch["observation.state"].shape)
    # print(ex_le_batch["next.done"])
    # print(ex_le_batch["timestamp"])
    # print(ex_le_batch["task"])

    # for key in ex_le_batch.keys():
    #     print(key, type(ex_le_batch[key]))

    # for key in example_batch["observation"]:
    #     print(key, type(example_batch["observation"][key]))

    # for key in example_batch["task"]:
    #     print(key, type(example_batch["task"][key]))

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    ###
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    # Fully override the old action head with a new one (for smaller changes, you can use update_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiscreteDiffusionActionHead,
        action_horizon=50,
        action_dim=14,
        readout_key="readout_action",
    )

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        ex_le_batch,
        text_processor,
        verbose=True,
        dataset_statistics=le_dataset_stats,
    )
    # model = OctoModel.from_config(
    #     config,
    #     example_batch,
    #     text_processor,
    #     verbose=True,
    #     dataset_statistics=le_dataset_stats,
    # )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(range(5000), total=5000, dynamic_ncols=True):
        batch = next(le_dataloader)
        # batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % 1000 == 0:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)
