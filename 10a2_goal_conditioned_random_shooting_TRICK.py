#10a2_goal_conditioned_random_shooting_TRICK.py

import argparse
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw

from transformers import AutoModel, AutoProcessor


# ============================================================
# 0. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 1. Tiny 1D Toy World (visual)
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

    # NEW VERSION WITH GRADIENT BACKGROUND
    def render(self) -> Image.Image:
        """
        Render current x as an image: a dot on a horizontal line,
        with a horizontal grayscale gradient background so I-JEPA
        sees stronger positional differences.
        """
        W = H = self.img_size
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        # Horizontal gradient background (left dark, right bright)
        for xpix in range(W):
            t = xpix / max(1, W - 1)    # 0..1
            val = int(255 * t)          # 0..255
            draw.line([(xpix, 0), (xpix, H)], fill=(val, val, val))

        # Map x in [-1, 1] to pixel coordinate
        left = self.margin
        right = W - self.margin
        t = (self.x + 1.0) / 2.0  # 0..1
        px = int(left + t * (right - left))
        py = H // 2

        # Draw central line (for visual reference)
        draw.line([(left, py), (right, py)], fill=(0, 0, 0), width=1)

        # Draw dot
        r = 3
        draw.ellipse(
            [(px - r, py - r), (px + r, py + r)],
            fill=(0, 0, 0),
        )

        return img



    # def render(self) -> Image.Image:
    #     """
    #     Render current x as an image: a dot on a horizontal line.
    #     """
    #     W = H = self.img_size
    #     img = Image.new("RGB", (W, H), color=(255, 255, 255))
    #     draw = ImageDraw.Draw(img)

    #     # Map x in [-1, 1] to pixel coordinate
    #     left = self.margin
    #     right = W - self.margin
    #     t = (self.x + 1.0) / 2.0  # 0..1
    #     px = int(left + t * (right - left))
    #     py = H // 2

    #     # Draw line
    #     draw.line([(left, py), (right, py)], fill=(0, 0, 0), width=1)

    #     # Draw dot
    #     r = 3
    #     draw.ellipse(
    #         [(px - r, py - r), (px + r, py + r)],
    #         fill=(0, 0, 0),
    #     )
    #     return img


# ============================================================
# 2. Frozen I-JEPA encoder
# ============================================================

def load_frozen_encoder(model_id: str = "facebook/ijepa_vith14_1k"):
    """
    Load frozen I-JEPA encoder + processor.
    Adjust model_id if needed.
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

    This uses CLS token from last_hidden_state[:, 0]
    and L2-normalizes it. Adjust if your earlier demos
    used a different pooling scheme.
    """
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # CLS token (batch, seq, dim) -> (batch, dim)
    z = outputs.last_hidden_state[:, 0]   # (1, D)
    z = F.normalize(z, dim=-1)
    return z.squeeze(0)                   # (D,)


# ============================================================
# 3. Latent dynamics predictor (same arch as training)
# ============================================================

class LatentDynamicsModel(nn.Module):
    """
    Simple MLP on [z, a] with residual output:
        z_next = normalize(z + f([z, a])).

    Make sure this matches the architecture you used to
    train your predictor.
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


def load_trained_predictor(
    ckpt_path: str,
    z_dim: int,
    action_dim: int,
) -> LatentDynamicsModel:
    """
    Load your previously trained predictor weights.
    ckpt_path must come from a model with the same architecture.
    """
    print(f"[Predictor] Loading checkpoint from {ckpt_path} ...")
    model = LatentDynamicsModel(z_dim=z_dim, action_dim=action_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("[Predictor] Loaded and set to eval().")
    return model


# ============================================================
# 4. Random-shooting planner in latent space
# ============================================================

def sample_action_sequences(
    num_candidates: int,
    horizon: int,
    action_dim: int,
    action_scale: float = 1.0,
) -> torch.Tensor:
    """
    Returns action sequences of shape (N, H, A).

    Here we use simple Gaussian actions ~ N(0, action_scale).
    For Toy1DWorld, action_dim=1 so it's just 1D actions.
    """
    actions = torch.randn(num_candidates, horizon, action_dim) * action_scale
    return actions.to(device)


@torch.no_grad()
def rollout_latent_sequences(
    z0: torch.Tensor,                    # (D,)
    actions: torch.Tensor,               # (N, H, A)
    predictor: LatentDynamicsModel,
) -> torch.Tensor:
    """
    Roll out all candidate action sequences in parallel.

    Returns:
        z_T: final latents of shape (N, D)
    """
    N, H, A = actions.shape
    D = z0.shape[-1]

    # Expand z0 to (N, D) to match candidates
    z = z0.unsqueeze(0).expand(N, D)

    for t in range(H):
        a_t = actions[:, t, :]          # (N, A)
        z = predictor(z, a_t)           # (N, D)

    return z


@torch.no_grad()
def score_candidates(
    z_final: torch.Tensor,   # (N, D)
    z_goal: torch.Tensor,    # (D,)
) -> torch.Tensor:
    """
    Cosine similarity between each candidate's final z_T and z_goal.
    Returns scores of shape (N,).
    """
    z_goal = z_goal.unsqueeze(0)        # (1, D)
    z_goal = F.normalize(z_goal, dim=-1)
    z_final = F.normalize(z_final, dim=-1)
    scores = (z_final * z_goal).sum(dim=-1)
    return scores


@torch.no_grad()
def random_shooting_plan(
    z_start: torch.Tensor,          # (D,)
    z_goal: torch.Tensor,           # (D,)
    predictor: LatentDynamicsModel,
    num_candidates: int,
    horizon: int,
    action_dim: int,
    action_scale: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    Run random shooting MPC in latent space.

    Returns:
        best_actions: (H, A)
        best_score: float (cosine similarity with goal)
    """
    # 1) Sample candidate sequences
    actions = sample_action_sequences(
        num_candidates=num_candidates,
        horizon=horizon,
        action_dim=action_dim,
        action_scale=action_scale,
    )   # (N, H, A)

    # 2) Rollout all sequences
    z_final = rollout_latent_sequences(z_start, actions, predictor)  # (N, D)

    # 3) Score
    scores = score_candidates(z_final, z_goal)                       # (N,)

    # 4) Pick best
    best_idx = torch.argmax(scores).item()
    best_actions = actions[best_idx]         # (H, A)
    best_score = scores[best_idx].item()

    return best_actions, best_score


# ============================================================
# 5. Demo: Toy world + I-JEPA + planner + GIF
# ============================================================

@torch.no_grad()
def demo_toy_world_planning(
    predictor: LatentDynamicsModel,
    encoder: AutoModel,
    processor: AutoProcessor,
    num_candidates: int,
    horizon: int,
    action_dim: int,
    action_scale: float,
    x_start: float,
    x_goal: float,
    out_gif: str,
):
    """
    Full demo:
    - Create Toy1DWorld
    - Render start & goal positions
    - Encode to latents with I-JEPA
    - Run random shooting in latent
    - Replay best plan in Toy1DWorld
    - Save a GIF
    """
    world = Toy1DWorld(img_size=64)

    # 1) Define start/goal positions
    print(f"[ToyWorld] Start x={x_start:.3f}, Goal x={x_goal:.3f}")

    # 2) Render start/goal images
    world.reset(x_start)
    img_start = world.render()

    world.reset(x_goal)
    img_goal = world.render()

    # 3) Encode to latents
    print("[ToyWorld] Encoding start/goal images to latents ...")
    z_start = encode_image_to_latent(img_start, encoder, processor)  # (D,)
    z_goal = encode_image_to_latent(img_goal, encoder, processor)    # (D,)

    # 4) Plan in latent space
    print("[Planner] Running goal-conditioned random shooting ...")
    best_actions, best_score = random_shooting_plan(
        z_start=z_start,
        z_goal=z_goal,
        predictor=predictor,
        num_candidates=num_candidates,
        horizon=horizon,
        action_dim=action_dim,
        action_scale=action_scale,
    )

    print(f"[Planner] Best score (cosine with z_goal): {best_score:.4f}")
    print("[Planner] Best action sequence (H x A):")
    print(best_actions.view(-1).cpu().numpy())

    # 5) Replay best plan in the real toy world and save frames
    print("[ToyWorld] Replaying best plan and saving frames ...")
    frames = []
    world.reset(x_start)
    frames.append(world.render())

    for t in range(horizon):
        a_t = best_actions[t].item()
        world.step(a_t, max_step=0.2)
        frames.append(world.render())

    # 6) Save as GIF for visual explanation
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    print(f"[ToyWorld] Saved visualization to {out_gif}")


# ============================================================
# 6. CLI + main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Demo 10: Goal-Conditioned Random Shooting (Toy1DWorld + I-JEPA)"
    )

    ap.add_argument(
        "--predictor_ckpt",
        type=str,
        required=True,
        help="Path to trained latent predictor checkpoint (.pt).",
    )

    ap.add_argument(
        "--model_id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="Hugging Face model id for I-JEPA encoder.",
    )

    ap.add_argument(
        "--z_dim",
        type=int,
        default=1280,
        help="Latent dimension D (must match encoder CLS dim + your predictor training).",
    )

    ap.add_argument(
        "--action_dim",
        type=int,
        default=1,
        help="Action dimension A (1 for Toy1DWorld).",
    )

    ap.add_argument(
        "--num_candidates",
        type=int,
        default=256,
        help="Number of random action sequences.",
    )

    ap.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Planning horizon (sequence length).",
    )

    ap.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Std dev for Gaussian action sampling.",
    )

    ap.add_argument(
        "--x_start",
        type=float,
        default=-0.6,
        help="Start position x in [-1, 1].",
    )

    ap.add_argument(
        "--x_goal",
        type=float,
        default=0.7,
        help="Goal position x in [-1, 1].",
    )

    ap.add_argument(
        "--out_gif",
        type=str,
        default="best_plan.gif",
        help="Path to output GIF.",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load encoder
    encoder, processor = load_frozen_encoder(args.model_id)

    # 2) Load predictor
    predictor = load_trained_predictor(
        ckpt_path=args.predictor_ckpt,
        z_dim=args.z_dim,
        action_dim=args.action_dim,
    )

    # 3) Run the toy world demo
    demo_toy_world_planning(
        predictor=predictor,
        encoder=encoder,
        processor=processor,
        num_candidates=args.num_candidates,
        horizon=args.horizon,
        action_dim=args.action_dim,
        action_scale=args.action_scale,
        x_start=args.x_start,
        x_goal=args.x_goal,
        out_gif=args.out_gif,
    )


if __name__ == "__main__":
    main()

