import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import DoublePendulumEnv

def train():
    # 1. Use argparse for command-line arguments (Requirement 6)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_type", type=str, default="shaped", choices=["baseline", "shaped"])
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--save_path", type=str, default="models/ppo_model.zip")
    args = parser.parse_args()

    # Create directories for models and logs
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 2. Instantiate the environment
    env = DoublePendulumEnv(reward_type=args.reward_type)
    
    # 3. Wrap the environment with Monitor for logging (Requirement 8)
    log_file = f"logs/monitor_{args.reward_type}.csv"
    env = Monitor(env, log_file)

    # 4. Instantiate the PPO agent (Requirement 6)
    # MLP Policy is suitable for our 6D vector observation space
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu"  # Using CPU for compatibility
    )

    print(f"Starting training: {args.reward_type} reward for {args.timesteps} steps...")
    
    # 5. Start the training
    model.learn(total_timesteps=args.timesteps)

    # 6. Save the final model (Requirement 6)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    train()
