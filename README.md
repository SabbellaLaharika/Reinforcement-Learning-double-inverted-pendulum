# Double Inverted Pendulum Reinforcement Learning

This project implements a custom 2D physics-based environment for a double inverted pendulum using `pymunk` and `pygame`. It includes a trained Proximal Policy Optimization (PPO) agent from `stable-baselines3`.

## Environment Design
The environment, `DoublePendulumEnv`, simulates a cart on a horizontal track with two interconnected poles.
- **Physics Engine**: `pymunk` handles rigid body dynamics, gravity, and constraints.
- **Constraints**: 
  - A `GrooveJoint` keeps the cart on a horizontal track.
  - `PivotJoints` connect the cart to the first pole, and the first pole to the second.
- **State Space**: A 6-dimensional vector: `[cart_x, cart_vx, pole1_angle, pole1_omega, pole2_angle, pole2_omega]`.
- **Action Space**: A single continuous value `[-1.0, 1.0]` representing the horizontal force applied to the cart.

## Reward Function Design

### Baseline Reward
The baseline reward focuses solely on the primary goal: keeping the poles upright.
`reward = cos(theta1) + cos(theta2)`
This gives a maximum reward of 2.0 when both poles are perfectly vertical.

### Shaped Reward
The shaped reward provides more frequent and detailed feedback to guide the agent's learning:
- **Upright Bonus**: `cos(theta1) + cos(theta2)` (same as baseline).
- **Center Penalty**: `-abs(cart_x) * 0.1` encourages the agent to stay near the center and avoid the boundaries.
- **Velocity Penalty**: `-(abs(omega1) + abs(omega2)) * 0.01` penalizes high angular velocities to encourage stability and prevent "jittery" behavior.
- **Energy/Action Penalty**: `-(action**2) * 0.001` encourages the agent to reach the goal using minimal force.

## How to Run

### 1. Build the Docker Image
Ensure you have Docker and Docker Compose installed.
```bash
docker-compose build
```

### 2. Train the Agent
To train the agent using the shaped reward function:
```bash
docker-compose run train
```
To compare with the baseline, you can override the command:
```bash
docker-compose run train python train.py --reward_type baseline --save_path models/baseline_model.zip
```

### 3. Evaluate the Agent
To visualize a trained model (requires an X-Server for GUI forwarding on Windows/host):
```bash
docker-compose run evaluate --model_path models/ppo_model.zip
```

### 4. Compare Results
Generate a plot comparing the learning curves of different rewards:
```bash
docker-compose run app python plot_results.py
```

## Results
- **Reward Comparison**: `reward_comparison.png` shows the training performance.
- **Performance**:
  - `media/agent_initial.gif`: Performance early in training.
  - `media/agent_final.gif`: Performance after full training.
