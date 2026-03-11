import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from environment import DoublePendulumEnv
import time

def evaluate():
    # 1. Use argparse for the model path (Requirement 7)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # 2. Instantiate the environment with rendering enabled
    env = DoublePendulumEnv(render_mode="human")
    
    # 3. Load the trained agent (Requirement 7)
    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path, env=env)

    # 4. Evaluation Loop
    obs, info = env.reset()
    print("Starting evaluation. Press Ctrl+C to stop.")
    
    try:
        for _ in range(2000): # Run for a decent length
            # Predict the action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render (Requirement 7)
            env.render()
            
            # Reset if the episode ends
            if terminated or truncated:
                obs, info = env.reset()
            
            # Small delay to make it smooth
            time.sleep(1/60)
            
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    evaluate()
