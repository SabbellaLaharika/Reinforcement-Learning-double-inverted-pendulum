import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pygame
import math

class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, reward_type='shaped', render_mode=None):
        super(DoublePendulumEnv, self).__init__()
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.dt = 1.0 / 60.0

        # Physical constants
        self.force_scale = 1000.0  # How hard the agent can push the cart
        self.pole_length = 100.0   # Length of each pole

        # 2a: Define Observation Space (Requirement 3)
        # Order: [cart_x, cart_vx, pole1_angle, pole1_omega, pole2_angle, pole2_omega]
        self.observation_space = spaces.Box(
            low=np.array([-500.0, -1000.0, -math.pi, -10.0, -math.pi, -10.0]),
            high=np.array([500.0, 1000.0, math.pi, 10.0, math.pi, 10.0]),
            dtype=np.float32
        )

        # 2a: Define Action Space (Requirement 3)
        # Continuous force between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 1.1 & 1.2: Initialize pygame and pymunk
        pygame.init()
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)  # Gravity pointing down

        self.screen = None
        self.clock = None

        # 1.5: Call self.reset() to create the initial physical objects
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Clear out old physics objects and create a new simulation
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)

        # 2. Create the cart Body and Shape
        self.cart_mass = 1.0
        inertia = pymunk.moment_for_box(self.cart_mass, (50, 30))
        self.cart_body = pymunk.Body(self.cart_mass, inertia)
        self.cart_body.position = (0, 0)
        self.cart_shape = pymunk.Poly.create_box(self.cart_body, (50, 30))

        # 3. Create the two pole Body and Shape objects
        self.pole1_mass = 0.5
        self.pole1_length = 100.0
        inertia1 = pymunk.moment_for_segment(self.pole1_mass, (0, 0), (0, -self.pole1_length), 5)
        self.pole1_body = pymunk.Body(self.pole1_mass, inertia1)
        self.pole1_body.position = (0, -15) # Start at top of cart
        self.pole1_shape = pymunk.Segment(self.pole1_body, (0, 0), (0, -self.pole1_length), 5)

        self.pole2_mass = 0.5
        self.pole2_length = 100.0
        inertia2 = pymunk.moment_for_segment(self.pole2_mass, (0, 0), (0, -self.pole2_length), 5)
        self.pole2_body = pymunk.Body(self.pole2_mass, inertia2)
        self.pole2_body.position = (0, -15 - self.pole1_length) # Start at top of pole 1
        self.pole2_shape = pymunk.Segment(self.pole2_body, (0, 0), (0, -self.pole2_length), 5)

        # 4. Create the joints
        # Constrain cart to horizontal track
        track = pymunk.GrooveJoint(self.space.static_body, self.cart_body, (-1000, 0), (1000, 0), (0, 0))
        # Cart <-> Pole 1
        joint1 = pymunk.PivotJoint(self.cart_body, self.pole1_body, (0, -15), (0, 0))
        # Pole 1 <-> Pole 2
        joint2 = pymunk.PivotJoint(self.pole1_body, self.pole2_body, (0, -self.pole1_length), (0, 0))

        # 5. Add all bodies, shapes, and joints to the pymunk.Space
        self.space.add(
            self.cart_body, self.cart_shape,
            self.pole1_body, self.pole1_shape,
            self.pole2_body, self.pole2_shape,
            track, joint1, joint2
        )

        # Add small random noise to start state (slightly upright)
        self.pole1_body.angle = self.np_random.uniform(-0.1, 0.1)
        self.pole2_body.angle = self.np_random.uniform(-0.1, 0.1)

        # 6. Return the initial observation
        return self._get_obs(), {}

    def _get_obs(self):
        # Extract values for our (6,) observation space
        return np.array([
            self.cart_body.position.x,
            self.cart_body.velocity.x,
            self.pole1_body.angle,
            self.pole1_body.angular_velocity,
            self.pole2_body.angle,
            self.pole2_body.angular_velocity
        ], dtype=np.float32)
            
    def _calculate_reward(self, obs, action):
        x, vx, theta1, omega1, theta2, omega2 = obs
        
        # Baseline Reward: cos(theta1) + cos(theta2) (Requirement 4)
        reward = math.cos(theta1) + math.cos(theta2)

        # Shaped Reward (Requirement 5)
        if self.reward_type == 'shaped':
            reward -= abs(x) * 0.1             # Center penalty
            reward -= (abs(omega1) + abs(omega2)) * 0.01  # Velocity penalty
            reward -= (action[0]**2) * 0.001   # Action penalty
            
        return reward

    def step(self, action):
        # 2. Apply force to cart
        force = action[0] * self.force_scale
        self.cart_body.apply_force_at_local_point((force, 0), (0, 0))

        # 3. Advance physics simulation
        self.space.step(self.dt)

        # 6. Gather new observation
        obs = self._get_obs()
        x, vx, theta1, omega1, theta2, omega2 = obs

        # 4. Calculate reward
        reward = self._calculate_reward(obs, action)

        # 5. Determine if the episode is over
        terminated = False
        if abs(x) > 400 or abs(theta1) > math.pi/2 or abs(theta2) > math.pi/2:
            terminated = True
            reward -= 10.0 # Penalty for falling

        truncated = False
        
        # 7. Return (observation, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((800, 600))
            else:
                self.screen = pygame.Surface((800, 600))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        
        # Offset to center of screen
        offset = np.array([400, 300])

        def to_pygame(p):
            return int(p[0] + offset[0]), int(offset[1] - p[1])

        # Draw components
        # Draw Cart
        pygame.draw.rect(self.screen, (100, 100, 100), (int(self.cart_body.position.x + 375), 285, 50, 30))

        # Draw Pole 1
        p1_start = to_pygame(self.pole1_body.position)
        p1_end = to_pygame(self.pole1_body.local_to_world((0, -self.pole1_length)))
        pygame.draw.line(self.screen, (200, 50, 50), p1_start, p1_end, 5)

        # Draw Pole 2
        p2_start = to_pygame(self.pole2_body.position)
        p2_end = to_pygame(self.pole2_body.local_to_world((0, -self.pole2_length)))
        pygame.draw.line(self.screen, (50, 50, 200), p2_start, p2_end, 5)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
