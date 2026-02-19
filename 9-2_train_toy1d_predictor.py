#9-2_train_toy1d_predictor.py  (RENAMED)
import argparse
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw

from transformers import AutoModel, AutoProcessor


# ============================================================
# 0. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 1. Tiny 1D Toy World (same idea as Demo 10)
# ============================================================

class Toy1DWorld:
    """
    Tiny 1D world: a point moves left/right on a horizontal line.
    State: scalar x in [-1, 1].
    """

    def __init__(self, img_size: int = 64, margin: int = 8):
        self.img_size = img_size
        self.margin = margin
        self.x = 0.0  # current position

    def reset(self, x: float) -> float:
        """Set the state to a given x in [-1, 1]."""
        x = float(x)
        self.x = max(-1.0, min(1.0, x))
        return self.x

    def step(self, action: float, max_step: float = 0.2) -> float:
        """
        Apply an action (delta-x) and clamp to [-1, 1].
        action is assumed to be roughly in [-1, 1],
        so we scale by max_step.
        """
        dx = float(action) * max_step
        self.x = max(-1.0, min(1.0, self.x + dx))
        return self.x

    def render(self) -> Image.Image:
        """
        Render current x as an image: a dot on a horizontal line.
        """
        W = H = self.img_size
        img = Image.new("RGB", (W, H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Map x in [-1, 1] to pixel coordinate
        left = self.margin
        right = W - self.margin
        t = (self.x + 1.0) / 2.0  # 0..1
        px = int(left + t * (right - left))
        py = H // 2

        # Draw line
        draw.line([(left, py), (right, py)], fill=(0, 0, 0), width=1)

        # Draw dot
        r = 3
        draw.ellipse(
            [(px - r, py - r), (px + r, py + r)],
            fill=(0, 0, 0),
        )
        return img


# ============================================================
# 2. Frozen I-JEPA encoder
# ============================================================

def load_frozen_encoder(model_id: str = "facebook/ijepa_vith14_1k"):
    """
    Load frozen I-JEPA encoder + processor.
    """
    print(f"[Encoder] Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[Encoder] Loaded and frozen.")
    return model, processor


@torch.no_grad()
def encode_image_to_latent(
    img: Image.Image,
    model: AutoModel,
    processor: AutoProcessor,
) -> torch.Tensor:
    """
    Return a single latent z of shape (D,) on device.

    Uses CLS token from last_hidden_state[:, 0] and L2-normalizes it.
    """
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)

    z = outputs.last_hidden_state[:, 0]   # (1, D)
    z = F.normalize(z, dim=-1)
    return z.squeeze(0)                   # (D,)


# ============================================================
# 3. Latent dynamics predictor (same arch as Demo 10)
# ============================================================

class LatentDynamicsModel(nn.Module):
    """
    MLP on [z, a] with residual output:
        z_next = normalize(z + f([z, a])).
    """

    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: (..., D)
        a: (..., A)
        returns: (..., D)
        """
        x = torch.cat([z, a], dim=-1)
        dz = self.net(x)
        z_next = z + dz
        z_next = F.normalize(z_next, dim=-1)
        return z_next


# ============================================================
# 4. Dataset generation: (z_t, a_t, z_{t+1})
# ============================================================

@torch.no_grad()
def generate_dataset(
    num_samples: int,
    encoder: AutoModel,
    processor: AutoProcessor,
    world: Toy1DWorld,
    action_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset of transitions in latent space.

    For each sample:
        - sample x_t ~ Uniform[-1, 1]
        - sample a_t ~ Uniform[-1, 1] (1D action)
        - x_{t+1} = step(x_t, a_t)
        - render images at t and t+1
        - encode both to latents z_t, z_{t+1}
    """
    print(f"[Data] Generating {num_samples} transitions ...")

    zs_t = []
    zs_tp1 = []
    actions = []

    for i in range(num_samples):
        # random start position
        x_t = random.uniform(-1.0, 1.0)
        world.reset(x_t)
        img_t = world.render()

        # random action (1D)
        a = random.uniform(-1.0, 1.0)
        world.step(a)  # updates internal x
        img_tp1 = world.render()

        # encode both images
        z_t = encode_image_to_latent(img_t, encoder, processor)      # (D,)
        z_tp1 = encode_image_to_latent(img_tp1, encoder, processor)  # (D,)

        zs_t.append(z_t.cpu())
        zs_tp1.append(z_tp1.cpu())

        a_vec = torch.zeros(action_dim)
        a_vec[0] = a
        actions.append(a_vec)

        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"[Data] {i+1}/{num_samples} done")

    zs_t = torch.stack(zs_t, dim=0)        # (N, D)
    zs_tp1 = torch.stack(zs_tp1, dim=0)    # (N, D)
    actions = torch.stack(actions, dim=0)  # (N, A)

    print("[Data] Shapes:",
          "z_t", zs_t.shape,
          "a_t", actions.shape,
          "z_tp1", zs_tp1.shape)

    return zs_t, actions, zs_tp1


# ============================================================
# 5. Training loop
# ============================================================

def train_predictor(
    zs_t: torch.Tensor,
    actions: torch.Tensor,
    zs_tp1: torch.Tensor,
    batch_size: int,
    num_epochs: int,
    lr: float,
    hidden_dim: int,
    ckpt_path: str,
):
    """
    Train LatentDynamicsModel on (z_t, a_t, z_{t+1}) with cosine loss.
    """
    dataset = TensorDataset(zs_t, actions, zs_tp1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    z_dim = zs_t.shape[-1]
    action_dim = actions.shape[-1]

    model = LatentDynamicsModel(z_dim=z_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print("[Train] Starting training ...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for z_t, a_t, z_tp1 in loader:
            z_t = z_t.to(device)
            a_t = a_t.to(device)
            z_tp1 = z_tp1.to(device)

            pred_z_tp1 = model(z_t, a_t)   # (B, D)

            # cosine similarity loss: 1 - cos(pred, target)
            cos_sim = F.cosine_similarity(pred_z_tp1, z_tp1, dim=-1)  # (B,)
            loss = (1.0 - cos_sim).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"[Train] Epoch {epoch:03d}/{num_epochs:03d}  loss={avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Train] Saved predictor checkpoint to: {ckpt_path}")


# ============================================================
# 6. CLI + main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train toy 1D latent dynamics predictor using I-JEPA latents."
    )

    ap.add_argument(
        "--out_ckpt",
        type=str,
        default="./checkpoints/predictor_toy1d.pt",
        help="Where to save the trained predictor checkpoint.",
    )

    ap.add_argument(
        "--model_id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="HF model id for I-JEPA encoder.",
    )

    ap.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of transitions to generate.",
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    ap.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    ap.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden dimension in the MLP.",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Setup toy world
    world = Toy1DWorld(img_size=64)

    # 2) Load frozen encoder
    encoder, processor = load_frozen_encoder(args.model_id)

    # 3) Generate dataset
    with torch.no_grad():
        zs_t, actions, zs_tp1 = generate_dataset(
            num_samples=args.num_samples,
            encoder=encoder,
            processor=processor,
            world=world,
            action_dim=1,
        )

    # 4) Train predictor
    train_predictor(
        zs_t=zs_t,
        actions=actions,
        zs_tp1=zs_tp1,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        ckpt_path=args.out_ckpt,
    )


if __name__ == "__main__":
    main()
