import sys
import os
import numpy as np
import pytest

# Add the parent directory to the python path so it can find environment.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DoublePendulumEnv
def test_environment_reset():
    """Test that the environment resets correctly and returns a valid initial state."""
    env = DoublePendulumEnv()
    obs, info = env.reset()
    
    assert obs.shape == (6,), "Observation space should have 6 dimensions"
    assert env.space is not None, "Pymunk space should be initialized"
    assert isinstance(info, dict), "Info should be a dictionary"
    
def test_environment_step():
    """Test that taking a step returns the correct types and formats."""
    env = DoublePendulumEnv()
    env.reset()
    
    # Take a zero action
    action = np.array([0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (6,), "Observation space should have 6 dimensions after step"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    
def test_observation_bounds():
    """Test that the initial observation is within the defined observation space bounds."""
    env = DoublePendulumEnv()
    obs, _ = env.reset()
    
    assert np.all(obs >= env.observation_space.low), "Observation out of lower bounds"
    assert np.all(obs <= env.observation_space.high), "Observation out of upper bounds"
