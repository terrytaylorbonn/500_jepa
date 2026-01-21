#12a_maniskill.py

import os

# Try to strongly discourage any GUI/Vulkan usage
os.environ["DISPLAY"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"
os.environ["SVULKAN2_HEADLESS"] = "1"     # sapient/svulkan2 hint (headless)

import gymnasium as gym
import mani_skill.envs  # ensure ManiSkill is imported
import numpy as np


def make_env():
    # Minimal, CPU, no-render env config
    env = gym.make(
        "PickCube-v1",              # adjust if your version uses different ids
        obs_mode="state",           # no images, just low-dim state
        control_mode="pd_joint_delta_pos",
        render_mode=None,           # no rendering
        sim_backend="cpu",          # CPU physics
        # NOTE: we drop shader_dir and num_envs for now
    )
    return env


def basic_rollout():
    env = make_env()
    obs, info = env.reset()   # gymnasium API: reset() -> (obs, info)

    done = False
    step_count = 0
    trunc = False

    while not (done or trunc) and step_count < 50:
        # Sample a random action from the env's action space
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        step_count += 1
        # Just to see something:
        # print(f"step={step_count}, reward={reward}")

    env.close()
    print("Rollout finished, steps:", step_count)


if __name__ == "__main__":
    basic_rollout()




# import os
# os.environ["DISPLAY"] = ""  # Disable display to avoid Vulkan issues
# os.environ["MUJOCO_GL"] = "osmesa"  # Use software rendering
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# import gymnasium as gym
# import mani_skill.envs  # or appropriate import for your installed version
# import numpy as np

# def make_env():
#     # Pick a simple task – you can change this to any supported env
#     # NOTE: actual env id might be slightly different in your version
#     env = gym.make(
#         "PickCube-v1",           # example: pick up a cube
#         obs_mode="state",        # use "state" instead of "rgbd" to avoid rendering issues
#         control_mode="pd_joint_delta_pos",  # simple joint control
#         render_mode=None,        # disable rendering to avoid Vulkan issues
#         num_envs=1,
#         sim_backend="cpu",       # use CPU backend to avoid Vulkan issues
#         shader_dir="minimal",    # use minimal shader to reduce GPU requirements
#     )
#     return env

# def basic_rollout():
#     env = make_env()
#     obs = env.reset()
#     done = False
#     step_count = 0

#     while not done and step_count < 50:
#         # Action: small random joint commands
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         step_count += 1

#     env.close()

# if __name__ == "__main__":
#     basic_rollout()
