"""
14b_train_latent_predictor_onehot.py
#106 New 14b: “encode once, then train on latents”

Two-stage pipeline:

1) Encode all Toy2D images from 14a with a frozen I-JEPA encoder
   into latents z_t, z_tp1 (one pass through the model).
2) Train a small latent dynamics model with one-hot actions:

      (z_t, a) -> z_hat_{t+1}

This avoids repeatedly calling JEPA inside the training loop.
"""

import os
import argparse
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoProcessor


# -------------------------------------------------------------
#  Latent predictor with one-hot actions
# -------------------------------------------------------------

class LatentPredictorOneHot(nn.Module):
    """
    Simple MLP dynamics model:

      input:  concat( z_t, one_hot(a_t) ) ∈ R^{z_dim + num_actions}
      output: z_hat_{t+1} ∈ R^{z_dim}
    """

    def __init__(self, z_dim: int, num_actions: int, hidden_dim: int = 1024):
        super().__init__()
        self.num_actions = num_actions

        self.fc1 = nn.Linear(z_dim + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, z_dim)

    def forward(self, z: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        """
        z:     (B, z_dim)
        a_idx: (B,) integer actions in [0, num_actions-1]
        """
        a_onehot = F.one_hot(a_idx, num_classes=self.num_actions).float()
        x = torch.cat([z, a_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z_next = self.fc_out(x)
        return z_next


# -------------------------------------------------------------
#  I-JEPA encoding helpers
# -------------------------------------------------------------

def load_ijepa_encoder(model_id: str, device: torch.device):
    print(f"[14b] Loading I-JEPA processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    print("[14b] Processor loaded.", flush=True)

    print(f"[14b] Loading I-JEPA encoder weights on {device}...", flush=True)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    print(f"[14b] Encoder loaded on {device}.", flush=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, processor


@torch.no_grad()
def encode_all_images(
    imgs_np: np.ndarray,
    encoder,
    processor,
    device: torch.device,
    batch_size: int = 32,
    name: str = "obs_t",
) -> torch.Tensor:
    """
    Encode ALL images in imgs_np into CLS latents.

    imgs_np: (N, H, W, 3) uint8
    Returns: (N, z_dim) CPU tensor.
    """
    N = imgs_np.shape[0]
    latents = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_arr = imgs_np[start:end]  # (B, H, W, 3)

        # Convert to list of PIL images
        batch_imgs = [Image.fromarray(arr) for arr in batch_arr]

        inputs = processor(batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = encoder(**inputs)
        z = outputs.last_hidden_state[:, 0, :]  # (B, z_dim)

        latents.append(z.cpu())  # keep latents on CPU
        print(f"[14b] Encoded {name} {end}/{N}", flush=True)

    z_all = torch.cat(latents, dim=0)  # (N, z_dim)
    return z_all


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-npz",
        type=str,
        required=True,
        help="Path to .npz file from 14a_toy2d_world_random_npz.py",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=4,
        help="Number of discrete actions in Toy2D.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="HuggingFace model id for I-JEPA.",
    )
    parser.add_argument(
        "--batch-size-encode",
        type=int,
        default=32,
        help="Batch size for JEPA encoding.",
    )
    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=64,
        help="Batch size for training predictor.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs for predictor.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--ckpt-out",
        type=str,
        default="./checkpoints/14b_predictor_2d_onehot_1k.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------------------------------------------
    # 1) Load 14a data
    # ---------------------------------------------------------
    print(f"[14b] Loading data from {args.data_npz}", flush=True)
    data = np.load(args.data_npz)
    obs_t_np = data["obs_t"]     # (N, H, W, 3) uint8
    obs_tp1_np = data["obs_tp1"]
    actions_np = data["actions"] # (N,) int64

    N = obs_t_np.shape[0]
    print(f"[14b] Dataset size: {N} transitions", flush=True)

    # ---------------------------------------------------------
    # 2) Encode ALL images once with JEPA
    # ---------------------------------------------------------
    # Use GPU if available for encoding to speed things up
    device_enc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[14b] Encoding on device: {device_enc}", flush=True)

    encoder, processor = load_ijepa_encoder(args.model_id, device_enc)

    # Encode obs_t and obs_tp1 → z_t_all, z_tp1_all (on CPU)
    z_t_all = encode_all_images(
        obs_t_np,
        encoder,
        processor,
        device=device_enc,
        batch_size=args.batch_size_encode,
        name="obs_t",
    )
    z_tp1_all = encode_all_images(
        obs_tp1_np,
        encoder,
        processor,
        device=device_enc,
        batch_size=args.batch_size_encode,
        name="obs_tp1",
    )

    # Free encoder to release GPU/CPU RAM
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Determine latent dimension from JEPA output
    z_dim = z_t_all.shape[1]
    print(f"[14b] Latent dimension z_dim={z_dim}", flush=True)

    # Actions tensor
    actions_all = torch.from_numpy(actions_np).long()  # (N,)

    # ---------------------------------------------------------
    # 3) Train latent predictor on latents only
    # ---------------------------------------------------------
    # Use GPU if available, but this is small so CPU also fine
    device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[14b] Training predictor on device: {device_train}", flush=True)

    predictor = LatentPredictorOneHot(
        z_dim=z_dim,
        num_actions=args.num_actions,
        hidden_dim=1024,
    ).to(device_train)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    N = z_t_all.shape[0]

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        epoch_loss = 0.0
        total = 0

        # Random permutation for SGD
        perm = torch.randperm(N)

        for start in range(0, N, args.batch_size_train):
            end = min(start + args.batch_size_train, N)
            idx = perm[start:end]

            z_t_batch = z_t_all[idx].to(device_train)      # (B, z_dim)
            z_tp1_batch = z_tp1_all[idx].to(device_train)  # (B, z_dim)
            a_batch = actions_all[idx].to(device_train)    # (B,)

            z_pred = predictor(z_t_batch, a_batch)

            cos_sim = F.cosine_similarity(z_pred, z_tp1_batch, dim=-1)
            loss = (1.0 - cos_sim).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = z_t_batch.size(0)
            epoch_loss += loss.item() * batch_size
            total += batch_size

        avg_loss = epoch_loss / max(total, 1)
        print(f"[14b] Epoch {epoch:03d} | loss={avg_loss:.4f}", flush=True)

    # ---------------------------------------------------------
    # 4) Save checkpoint
    # ---------------------------------------------------------
    ckpt = {
        "predictor": predictor.state_dict(),
        "z_dim": z_dim,
        "num_actions": args.num_actions,
        "model_id": args.model_id,
    }
    torch.save(ckpt, args.ckpt_out)
    print(f"[14b] Saved checkpoint to {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()



# """
# 14b_train_latent_predictor_onehot.py (CPU-only) #105 force GPU

# Train a latent dynamics model on Toy2D transitions (from 14a),
# using:

# - Frozen I-JEPA encoder (facebook/ijepa_vith14_1k)
# - One-hot actions (no separate action embedding)
# - Loss: 1 - cosine_similarity(z_hat_{t+1}, z_{t+1})

# Runs entirely on CPU to avoid GPU/VRAM issues.

# Input data is produced by 14a_toy2d_world_random_npz.py and expected to have:

#   obs_t   : (N, H, W, 3) uint8
#   obs_tp1 : (N, H, W, 3) uint8
#   actions : (N,) int64

# Usage:

#   python 14b_train_latent_predictor_onehot.py \
#     --data-npz ./data/toy2d_14a_data.npz \
#     --z-dim 1280 \
#     --num-actions 4 \
#     --epochs 10 \
#     --ckpt-out ./checkpoints/14b_predictor_2d_onehot_1k.pt
# """

# import os
# import argparse
# from typing import List, Tuple

# import numpy as np
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# from transformers import AutoModel, AutoProcessor


# # -------------------------------------------------------------
# #  Dataset: wrap 14a .npz
# # -------------------------------------------------------------

# class Toy2DTransitions14a(Dataset):
#     """
#     Wraps the data produced by 14a_toy2d_world_random_npz.py.

#     Expected keys in npz:
#       - obs_t   : (N, H, W, 3) uint8
#       - obs_tp1 : (N, H, W, 3) uint8
#       - actions : (N,) int
#     """

#     def __init__(self, npz_path: str):
#         data = np.load(npz_path)
#         self.obs_t = data["obs_t"]       # np.ndarray
#         self.obs_tp1 = data["obs_tp1"]
#         self.actions = data["actions"]

#     def __len__(self) -> int:
#         return len(self.actions)

#     def __getitem__(self, idx: int):
#         x_t_arr = self.obs_t[idx]        # (H, W, 3) uint8
#         x_tp1_arr = self.obs_tp1[idx]
#         a = int(self.actions[idx])
#         return x_t_arr, x_tp1_arr, a


# # -------------------------------------------------------------
# #  I-JEPA encoder loader (CPU-only)
# # -------------------------------------------------------------

# def load_ijepa_encoder_cpu(model_id: str):
#     """
#     Load a frozen I-JEPA encoder + processor on CPU.
#     """
#     print(f"[14b] Loading I-JEPA processor on CPU: {model_id}", flush=True)
#     processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
#     print("[14b] Processor loaded.", flush=True)

#     print(f"[14b] Loading I-JEPA encoder weights on CPU...", flush=True)
#     encoder = AutoModel.from_pretrained(model_id)  # stays on CPU
#     print("[14b] Encoder loaded on CPU.", flush=True)

#     encoder.eval()
#     for p in encoder.parameters():
#         p.requires_grad_(False)

#     return encoder, processor


# @torch.no_grad()
# def encode_batch_images_cpu(
#     imgs: List[Image.Image],
#     encoder,
#     processor,
# ) -> torch.Tensor:
#     """
#     Encode a list of PIL images into CLS latents using CPU.

#     Returns: (B, z_dim) on CPU.
#     """
#     inputs = processor(imgs, return_tensors="pt")
#     outputs = encoder(**inputs)
#     z = outputs.last_hidden_state[:, 0, :]  # CLS embedding
#     return z  # CPU tensor


# # -------------------------------------------------------------
# #  Latent predictor with one-hot actions (CPU)
# # -------------------------------------------------------------

# class LatentPredictorOneHot(nn.Module):
#     """
#     Simple MLP dynamics model:

#       input:  concat( z_t, one_hot(a_t) ) ∈ R^{z_dim + num_actions}
#       output: z_hat_{t+1} ∈ R^{z_dim}
#     """

#     def __init__(self, z_dim: int, num_actions: int, hidden_dim: int = 1024):
#         super().__init__()
#         self.num_actions = num_actions

#         self.fc1 = nn.Linear(z_dim + num_actions, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc_out = nn.Linear(hidden_dim, z_dim)

#     def forward(self, z: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
#         """
#         z:     (B, z_dim) CPU tensor
#         a_idx: (B,) integer actions in [0, num_actions-1] (CPU)
#         """
#         a_onehot = F.one_hot(a_idx, num_classes=self.num_actions).float()
#         x = torch.cat([z, a_onehot], dim=-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         z_next = self.fc_out(x)
#         return z_next


# # -------------------------------------------------------------
# #  Custom collate function (keep arrays as lists)
# # -------------------------------------------------------------

# def collate_npimg_batch(batch):
#     """
#     batch is a list of (x_t_arr, x_tp1_arr, a)

#     We want:
#       x_t_list      : list of np.ndarray (H, W, 3)
#       x_tp1_list    : list of np.ndarray
#       a_idx_list    : list of ints
#     """
#     x_t_list, x_tp1_list, a_list = zip(*batch)
#     return list(x_t_list), list(x_tp1_list), list(a_list)


# # -------------------------------------------------------------
# #  Training loop (CPU)
# # -------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data-npz",
#         type=str,
#         required=True,
#         help="Path to .npz file from 14a_toy2d_world_random_npz.py",
#     )
#     parser.add_argument(
#         "--z-dim",
#         type=int,
#         default=1280,
#         help="Latent dimension (CLS) of I-JEPA.",
#     )
#     parser.add_argument(
#         "--num-actions",
#         type=int,
#         default=4,
#         help="Number of discrete actions in Toy2D.",
#     )
#     parser.add_argument(
#         "--model-id",
#         type=str,
#         default="facebook/ijepa_vith14_1k",
#         help="HuggingFace model id for I-JEPA.",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=64,
#         help="Batch size for training.",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=10,
#         help="Number of training epochs.",
#     )
#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=1e-3,
#         help="Learning rate for Adam.",
#     )
#     parser.add_argument(
#         "--ckpt-out",
#         type=str,
#         default="./checkpoints/14b_predictor_2d_onehot_1k.pt",
#         help="Output checkpoint path.",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=0,
#         help="Random seed.",
#     )
#     args = parser.parse_args()

#     os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)

#     print("[14b] Forcing CPU-only training (no CUDA).", flush=True)

#     # 1) Load frozen I-JEPA encoder (CPU)
#     encoder, processor = load_ijepa_encoder_cpu(args.model_id)

#     # 2) Dataset + loader (with custom collate)
#     dataset = Toy2DTransitions14a(args.data_npz)
#     loader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         collate_fn=collate_npimg_batch,
#     )

#     # 3) Predictor model on CPU
#     predictor = LatentPredictorOneHot(
#         z_dim=args.z_dim,
#         num_actions=args.num_actions,
#         hidden_dim=1024,
#     )
#     predictor.train()

#     optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

#     # 4) Training loop
#     for epoch in range(1, args.epochs + 1):
#         predictor.train()
#         epoch_loss = 0.0
#         total = 0

#         for x_t_list, x_tp1_list, a_idx_list in loader:
#             # Convert numpy arrays -> PIL images
#             x_t_imgs = [Image.fromarray(arr) for arr in x_t_list]
#             x_tp1_imgs = [Image.fromarray(arr) for arr in x_tp1_list]

#             # Actions as CPU tensor
#             a_idx = torch.tensor(a_idx_list, dtype=torch.long)  # CPU

#             # Encode obs_t and obs_tp1 into latents (CPU)
#             with torch.no_grad():
#                 z_t = encode_batch_images_cpu(x_t_imgs, encoder, processor)      # CPU
#                 z_tp1 = encode_batch_images_cpu(x_tp1_imgs, encoder, processor)  # CPU

#             # Predictor forward (CPU)
#             z_pred = predictor(z_t, a_idx)

#             # Loss: 1 - cosine_similarity
#             cos_sim = F.cosine_similarity(z_pred, z_tp1, dim=-1)
#             loss = (1.0 - cos_sim).mean()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             batch_size = z_t.size(0)
#             epoch_loss += loss.item() * batch_size
#             total += batch_size

#         avg_loss = epoch_loss / max(total, 1)
#         print(f"[14b] Epoch {epoch:03d} | loss={avg_loss:.4f}", flush=True)

#     # 5) Save checkpoint (CPU weights; still fine)
#     ckpt = {
#         "predictor": predictor.state_dict(),
#         "z_dim": args.z_dim,
#         "num_actions": args.num_actions,
#         "model_id": args.model_id,
#     }
#     torch.save(ckpt, args.ckpt_out)
#     print(f"[14b] Saved checkpoint to {args.ckpt_out}", flush=True)


# if __name__ == "__main__":
#     main()



# # """
# # 14b_train_latent_predictor_onehot.py (#104 error fix)

# # Train a latent dynamics model on Toy2D transitions (from 14a),
# # using:

# # - Frozen I-JEPA encoder (facebook/ijepa_vith14_1k)
# # - One-hot actions (no separate action embedding)
# # - Loss: 1 - cosine_similarity(z_hat_{t+1}, z_{t+1})

# # Input data is produced by 14a_toy2d_world_random_npz.py and expected to have:

# #   obs_t   : (N, H, W, 3) uint8
# #   obs_tp1 : (N, H, W, 3) uint8
# #   actions : (N,) int64

# # Usage:

# #   python 14b_train_latent_predictor_onehot.py \
# #     --data-npz ./data/toy2d_14a_data.npz \
# #     --z-dim 1280 \
# #     --num-actions 4 \
# #     --epochs 20 \
# #     --ckpt-out ./checkpoints/14b_predictor_2d_onehot_1k.pt
# # """

# # import os
# # import argparse
# # from typing import List, Tuple

# # import numpy as np
# # from PIL import Image

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader

# # from transformers import AutoModel, AutoProcessor


# # # -------------------------------------------------------------
# # #  Dataset: wrap 14a .npz
# # # -------------------------------------------------------------

# # class Toy2DTransitions14a(Dataset):
# #     """
# #     Wraps the data produced by 14a_toy2d_world_random_npz.py.

# #     Expected keys in npz:
# #       - obs_t   : (N, H, W, 3) uint8
# #       - obs_tp1 : (N, H, W, 3) uint8
# #       - actions : (N,) int
# #     """

# #     def __init__(self, npz_path: str):
# #         data = np.load(npz_path)
# #         self.obs_t = data["obs_t"]       # np.ndarray
# #         self.obs_tp1 = data["obs_tp1"]
# #         self.actions = data["actions"]

# #     def __len__(self) -> int:
# #         return len(self.actions)

# #     def __getitem__(self, idx: int):
# #         # Return raw numpy arrays, not PIL. We'll convert later.
# #         x_t_arr = self.obs_t[idx]        # (H, W, 3) uint8
# #         x_tp1_arr = self.obs_tp1[idx]
# #         a = int(self.actions[idx])
# #         return x_t_arr, x_tp1_arr, a


# # # -------------------------------------------------------------
# # #  I-JEPA encoder loader
# # # -------------------------------------------------------------

# # def load_ijepa_encoder(model_id: str, device: torch.device):
# #     """
# #     Load a frozen I-JEPA encoder + processor.
# #     """
# #     print(f"[14b] Loading I-JEPA model: {model_id}")
# #     processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
# #     encoder = AutoModel.from_pretrained(model_id).to(device)
# #     encoder.eval()
# #     for p in encoder.parameters():
# #         p.requires_grad_(False)
# #     return encoder, processor


# # @torch.no_grad()
# # def encode_batch_images(
# #     imgs: List[Image.Image],
# #     encoder,
# #     processor,
# #     device: torch.device,
# # ) -> torch.Tensor:
# #     """
# #     Encode a list of PIL images into CLS latents.

# #     Returns: (B, z_dim)
# #     """
# #     inputs = processor(imgs, return_tensors="pt")
# #     inputs = {k: v.to(device) for k, v in inputs.items()}
# #     outputs = encoder(**inputs)
# #     # CLS embedding
# #     z = outputs.last_hidden_state[:, 0, :]
# #     return z


# # def get_device() -> torch.device:
# #     if torch.cuda.is_available():
# #         print("[14b] Device: cuda")
# #         return torch.device("cuda")
# #     else:
# #         print("[14b] Device: cpu")
# #         return torch.device("cpu")


# # # -------------------------------------------------------------
# # #  Latent predictor with one-hot actions
# # # -------------------------------------------------------------

# # class LatentPredictorOneHot(nn.Module):
# #     """
# #     Simple MLP dynamics model:

# #       input:  concat( z_t, one_hot(a_t) ) ∈ R^{z_dim + num_actions}
# #       output: z_hat_{t+1} ∈ R^{z_dim}
# #     """

# #     def __init__(self, z_dim: int, num_actions: int, hidden_dim: int = 1024):
# #         super().__init__()
# #         self.num_actions = num_actions

# #         self.fc1 = nn.Linear(z_dim + num_actions, hidden_dim)
# #         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
# #         self.fc_out = nn.Linear(hidden_dim, z_dim)

# #     def forward(self, z: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
# #         """
# #         z:     (B, z_dim)
# #         a_idx: (B,) integer actions in [0, num_actions-1]
# #         """
# #         a_onehot = F.one_hot(a_idx, num_classes=self.num_actions).float()
# #         x = torch.cat([z, a_onehot], dim=-1)
# #         x = F.relu(self.fc1(x))
# #         x = F.relu(self.fc2(x))
# #         z_next = self.fc_out(x)
# #         return z_next


# # # -------------------------------------------------------------
# # #  Custom collate function (keep arrays as lists)
# # # -------------------------------------------------------------

# # def collate_npimg_batch(batch):
# #     """
# #     batch is a list of (x_t_arr, x_tp1_arr, a)

# #     We want:
# #       x_t_list      : list of np.ndarray (H, W, 3)
# #       x_tp1_list    : list of np.ndarray
# #       a_idx_list    : list of ints
# #     """
# #     x_t_list, x_tp1_list, a_list = zip(*batch)
# #     return list(x_t_list), list(x_tp1_list), list(a_list)


# # # -------------------------------------------------------------
# # #  Training loop
# # # -------------------------------------------------------------

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument(
# #         "--data-npz",
# #         type=str,
# #         required=True,
# #         help="Path to .npz file from 14a_toy2d_world_random_npz.py",
# #     )
# #     parser.add_argument(
# #         "--z-dim",
# #         type=int,
# #         default=1280,
# #         help="Latent dimension (CLS) of I-JEPA.",
# #     )
# #     parser.add_argument(
# #         "--num-actions",
# #         type=int,
# #         default=4,
# #         help="Number of discrete actions in Toy2D.",
# #     )
# #     parser.add_argument(
# #         "--model-id",
# #         type=str,
# #         default="facebook/ijepa_vith14_1k",
# #         help="HuggingFace model id for I-JEPA.",
# #     )
# #     parser.add_argument(
# #         "--batch-size",
# #         type=int,
# #         default=64,
# #         help="Batch size for training.",
# #     )
# #     parser.add_argument(
# #         "--epochs",
# #         type=int,
# #         default=20,
# #         help="Number of training epochs.",
# #     )
# #     parser.add_argument(
# #         "--lr",
# #         type=float,
# #         default=1e-3,
# #         help="Learning rate for Adam.",
# #     )
# #     parser.add_argument(
# #         "--ckpt-out",
# #         type=str,
# #         default="./checkpoints/14b_predictor_2d_onehot_1k.pt",
# #         help="Output checkpoint path.",
# #     )
# #     parser.add_argument(
# #         "--seed",
# #         type=int,
# #         default=0,
# #         help="Random seed.",
# #     )
# #     args = parser.parse_args()

# #     os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

# #     torch.manual_seed(args.seed)
# #     np.random.seed(args.seed)

# #     device = get_device()

# #     # 1) Load frozen I-JEPA encoder
# #     encoder, processor = load_ijepa_encoder(args.model_id, device=device)

# #     # 2) Dataset + loader (with custom collate)
# #     dataset = Toy2DTransitions14a(args.data_npz)
# #     loader = DataLoader(
# #         dataset,
# #         batch_size=args.batch_size,
# #         shuffle=True,
# #         collate_fn=collate_npimg_batch,
# #     )

# #     # 3) Predictor model
# #     predictor = LatentPredictorOneHot(
# #         z_dim=args.z_dim,
# #         num_actions=args.num_actions,
# #         hidden_dim=1024,
# #     ).to(device)

# #     optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

# #     # 4) Training loop
# #     for epoch in range(1, args.epochs + 1):
# #         predictor.train()
# #         epoch_loss = 0.0
# #         total = 0

# #         for x_t_list, x_tp1_list, a_idx_list in loader:
# #             # Convert numpy arrays -> PIL images
# #             x_t_imgs = [Image.fromarray(arr) for arr in x_t_list]
# #             x_tp1_imgs = [Image.fromarray(arr) for arr in x_tp1_list]

# #             a_idx = torch.tensor(a_idx_list, dtype=torch.long, device=device)

# #             # Encode obs_t and obs_tp1 into latents
# #             with torch.no_grad():
# #                 z_t = encode_batch_images(x_t_imgs, encoder, processor, device)
# #                 z_tp1 = encode_batch_images(x_tp1_imgs, encoder, processor, device)

# #             z_t = z_t.to(device)
# #             z_tp1 = z_tp1.to(device)

# #             # Predictor forward
# #             z_pred = predictor(z_t, a_idx)

# #             # Loss: 1 - cosine_similarity
# #             cos_sim = F.cosine_similarity(z_pred, z_tp1, dim=-1)
# #             loss = (1.0 - cos_sim).mean()

# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()

# #             batch_size = z_t.size(0)
# #             epoch_loss += loss.item() * batch_size
# #             total += batch_size

# #         avg_loss = epoch_loss / max(total, 1)
# #         print(f"[14b] Epoch {epoch:03d} | loss={avg_loss:.4f}")

# #     # 5) Save checkpoint
# #     ckpt = {
# #         "predictor": predictor.state_dict(),
# #         "z_dim": args.z_dim,
# #         "num_actions": args.num_actions,
# #         "model_id": args.model_id,
# #     }
# #     torch.save(ckpt, args.ckpt_out)
# #     print(f"[14b] Saved checkpoint to {args.ckpt_out}")


# # if __name__ == "__main__":
# #     main()



# # # """
# # # 14b_train_latent_predictor_onehot.py

# # # Train a latent dynamics model on Toy2D transitions (from 14a),
# # # using:

# # # - Frozen I-JEPA encoder (facebook/ijepa_vith14_1k)
# # # - One-hot actions (no separate action embedding)
# # # - Loss: 1 - cosine_similarity(z_hat_{t+1}, z_{t+1})

# # # Input data is produced by 14a_toy2d_world_random_npz.py and expected to have:

# # #   obs_t   : (N, H, W, 3) uint8
# # #   obs_tp1 : (N, H, W, 3) uint8
# # #   actions : (N,) int64

# # # Usage example:

# # #   python 14b_train_latent_predictor_onehot.py \
# # #     --data-npz ./data/toy2d_14a_data.npz \
# # #     --z-dim 1280 \
# # #     --num-actions 4 \
# # #     --epochs 20 \
# # #     --ckpt-out ./checkpoints/14b_predictor_2d_onehot_1k.pt
# # # """

# # # import os
# # # import argparse
# # # from typing import List, Tuple

# # # import numpy as np
# # # from PIL import Image

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from torch.utils.data import Dataset, DataLoader

# # # from transformers import AutoModel, AutoProcessor


# # # # -------------------------------------------------------------
# # # #  Dataset: wrap 14a .npz
# # # # -------------------------------------------------------------

# # # class Toy2DTransitions14a(Dataset):
# # #     """
# # #     Wraps the data produced by 14a_toy2d_world_random_npz.py.

# # #     Expected keys in npz:
# # #       - obs_t   : (N, H, W, 3) uint8
# # #       - obs_tp1 : (N, H, W, 3) uint8
# # #       - actions : (N,) int
# # #     """

# # #     def __init__(self, npz_path: str):
# # #         data = np.load(npz_path)
# # #         self.obs_t = data["obs_t"]
# # #         self.obs_tp1 = data["obs_tp1"]
# # #         self.actions = data["actions"]

# # #     def __len__(self) -> int:
# # #         return len(self.actions)

# # #     def __getitem__(self, idx: int):
# # #         x_t_arr = self.obs_t[idx]       # (H, W, 3) uint8
# # #         x_tp1_arr = self.obs_tp1[idx]
# # #         a = int(self.actions[idx])

# # #         # Convert to PIL for the JEPA processor
# # #         x_t_img = Image.fromarray(x_t_arr)
# # #         x_tp1_img = Image.fromarray(x_tp1_arr)
# # #         return x_t_img, x_tp1_img, a


# # # # -------------------------------------------------------------
# # # #  I-JEPA encoder loader
# # # # -------------------------------------------------------------

# # # def load_ijepa_encoder(model_id: str, device: torch.device):
# # #     """
# # #     Load a frozen I-JEPA encoder + processor.
# # #     """
# # #     print(f"[14b] Loading I-JEPA model: {model_id}")
# # #     processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
# # #     encoder = AutoModel.from_pretrained(model_id).to(device)
# # #     encoder.eval()
# # #     for p in encoder.parameters():
# # #         p.requires_grad_(False)
# # #     return encoder, processor


# # # @torch.no_grad()
# # # def encode_batch_images(
# # #     imgs: List[Image.Image],
# # #     encoder,
# # #     processor,
# # #     device: torch.device,
# # # ) -> torch.Tensor:
# # #     """
# # #     Encode a list of PIL images into CLS latents.

# # #     Returns: (B, z_dim)
# # #     """
# # #     inputs = processor(imgs, return_tensors="pt")
# # #     inputs = {k: v.to(device) for k, v in inputs.items()}
# # #     outputs = encoder(**inputs)
# # #     # CLS embedding
# # #     z = outputs.last_hidden_state[:, 0, :]
# # #     return z


# # # def get_device() -> torch.device:
# # #     if torch.cuda.is_available():
# # #         print("[14b] Device: cuda")
# # #         return torch.device("cuda")
# # #     else:
# # #         print("[14b] Device: cpu")
# # #         return torch.device("cpu")


# # # # -------------------------------------------------------------
# # # #  Latent predictor with one-hot actions
# # # # -------------------------------------------------------------

# # # class LatentPredictorOneHot(nn.Module):
# # #     """
# # #     Simple MLP dynamics model:

# # #       input:  concat( z_t, one_hot(a_t) ) ∈ R^{z_dim + num_actions}
# # #       output: z_hat_{t+1} ∈ R^{z_dim}
# # #     """

# # #     def __init__(self, z_dim: int, num_actions: int, hidden_dim: int = 1024):
# # #         super().__init__()
# # #         self.num_actions = num_actions

# # #         self.fc1 = nn.Linear(z_dim + num_actions, hidden_dim)
# # #         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
# # #         self.fc_out = nn.Linear(hidden_dim, z_dim)

# # #     def forward(self, z: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
# # #         """
# # #         z:     (B, z_dim)
# # #         a_idx: (B,) integer actions in [0, num_actions-1]
# # #         """
# # #         a_onehot = F.one_hot(a_idx, num_classes=self.num_actions).float()
# # #         x = torch.cat([z, a_onehot], dim=-1)
# # #         x = F.relu(self.fc1(x))
# # #         x = F.relu(self.fc2(x))
# # #         z_next = self.fc_out(x)
# # #         return z_next


# # # # -------------------------------------------------------------
# # # #  Training loop
# # # # -------------------------------------------------------------

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument(
# # #         "--data-npz",
# # #         type=str,
# # #         required=True,
# # #         help="Path to .npz file from 14a_toy2d_world_random_npz.py",
# # #     )
# # #     parser.add_argument(
# # #         "--z-dim",
# # #         type=int,
# # #         default=1280,
# # #         help="Latent dimension (CLS) of I-JEPA.",
# # #     )
# # #     parser.add_argument(
# # #         "--num-actions",
# # #         type=int,
# # #         default=4,
# # #         help="Number of discrete actions in Toy2D.",
# # #     )
# # #     parser.add_argument(
# # #         "--model-id",
# # #         type=str,
# # #         default="facebook/ijepa_vith14_1k",
# # #         help="HuggingFace model id for I-JEPA.",
# # #     )
# # #     parser.add_argument(
# # #         "--batch-size",
# # #         type=int,
# # #         default=64,
# # #         help="Batch size for training.",
# # #     )
# # #     parser.add_argument(
# # #         "--epochs",
# # #         type=int,
# # #         default=20,
# # #         help="Number of training epochs.",
# # #     )
# # #     parser.add_argument(
# # #         "--lr",
# # #         type=float,
# # #         default=1e-3,
# # #         help="Learning rate for Adam.",
# # #     )
# # #     parser.add_argument(
# # #         "--ckpt-out",
# # #         type=str,
# # #         default="./checkpoints/14b_predictor_2d_onehot_1k.pt",
# # #         help="Output checkpoint path.",
# # #     )
# # #     parser.add_argument(
# # #         "--seed",
# # #         type=int,
# # #         default=0,
# # #         help="Random seed.",
# # #     )
# # #     args = parser.parse_args()

# # #     os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

# # #     torch.manual_seed(args.seed)
# # #     np.random.seed(args.seed)

# # #     device = get_device()

# # #     # 1) Load frozen I-JEPA encoder
# # #     encoder, processor = load_ijepa_encoder(args.model_id, device=device)

# # #     # 2) Dataset + loader
# # #     dataset = Toy2DTransitions14a(args.data_npz if hasattr(args, "data_npz") else args.data_npz)
# # #     # ^ above is defensive; but argparse key is data-npz, which maps to args.data_npz
# # #     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# # #     # 3) Predictor model
# # #     predictor = LatentPredictorOneHot(
# # #         z_dim=args.z_dim,
# # #         num_actions=args.num_actions,
# # #         hidden_dim=1024,
# # #     ).to(device)

# # #     optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

# # #     # 4) Training loop
# # #     for epoch in range(1, args.epochs + 1):
# # #         predictor.train()
# # #         epoch_loss = 0.0
# # #         total = 0

# # #         for x_t_imgs, x_tp1_imgs, a_idx_list in loader:
# # #             # actions
# # #             a_idx = torch.tensor(a_idx_list, dtype=torch.long, device=device)

# # #             # Encode obs_t and obs_tp1 into latents
# # #             with torch.no_grad():
# # #                 z_t = encode_batch_images(list(x_t_imgs), encoder, processor, device)
# # #                 z_tp1 = encode_batch_images(list(x_tp1_imgs), encoder, processor, device)

# # #             z_t = z_t.to(device)
# # #             z_tp1 = z_tp1.to(device)

# # #             # Predictor forward
# # #             z_pred = predictor(z_t, a_idx)

# # #             # Loss: 1 - cosine_similarity
# # #             cos_sim = F.cosine_similarity(z_pred, z_tp1, dim=-1)
# # #             loss = (1.0 - cos_sim).mean()

# # #             optimizer.zero_grad()
# # #             loss.backward()
# # #             optimizer.step()

# # #             batch_size = z_t.size(0)
# # #             epoch_loss += loss.item() * batch_size
# # #             total += batch_size

# # #         avg_loss = epoch_loss / max(total, 1)
# # #         print(f"[14b] Epoch {epoch:03d} | loss={avg_loss:.4f}")

# # #     # 5) Save checkpoint
# # #     ckpt = {
# # #         "predictor": predictor.state_dict(),
# # #         "z_dim": args.z_dim,
# # #         "num_actions": args.num_actions,
# # #         "model_id": args.model_id,
# # #     }
# # #     torch.save(ckpt, args.ckpt_out)
# # #     print(f"[14b] Saved checkpoint to {args.ckpt_out}")


# # # if __name__ == "__main__":
# # #     main()
