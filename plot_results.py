import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_results():
    # 1. Use argparse for the logs directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="logs")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    # 2. Find all monitor CSV files (Requirement 8)
    if not os.path.exists(args.logs_dir):
        print(f"Directory {args.logs_dir} does not exist. Run training first.")
        return

    files = [f for f in os.listdir(args.logs_dir) if f.endswith(".csv")]
    
    for file in files:
        path = os.path.join(args.logs_dir, file)
        try:
            # stable-baselines3 Monitor files have a comment on the first line
            df = pd.read_csv(path, skiprows=1)
            
            # 'r' is the column for reward in SB3 Monitor logs
            if 'r' in df.columns:
                # Calculate mean reward over a window for smoothing
                df['mean_reward'] = df['r'].rolling(window=50).mean()
                label = file.replace("monitor_", "").replace(".csv", "")
                
                # 3. Plot Mean Reward vs Timesteps/Episodes (Requirement 9)
                plt.plot(df.index, df['mean_reward'], label=f"{label} reward")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    plt.title("Learning Curves: Reward Comparison")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot (Requirement 9)
    plt.savefig("reward_comparison.png")
    print("Plot saved to reward_comparison.png")

if __name__ == "__main__":
    plot_results()
