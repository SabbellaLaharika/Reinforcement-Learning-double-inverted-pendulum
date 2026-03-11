import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from environment import DoublePendulumEnv
import imageio
import numpy as np

def generate_gif():
    # 1. Use argparse for the model and output path (Requirement 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    # Create directories
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 2. Instantiate environment with rgb_array mode for GIF capture
    env = DoublePendulumEnv(render_mode="rgb_array")
    
    # 3. Load the model
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path)

    images = []
    obs, info = env.reset()
    
    print(f"Recording simulation for {args.steps} steps...")
    for _ in range(args.steps):
        # Capture the frame
        frame = env.render()
        images.append(frame)
        
        # Step the agent
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()

    # 4. Save as GIF (Requirement 10)
    print(f"Saving GIF to {args.output_path}...")
    imageio.mimsave(args.output_path, [np.array(img) for img in images], fps=60)
    print("Done.")

    env.close()

if __name__ == "__main__":
    generate_gif()
