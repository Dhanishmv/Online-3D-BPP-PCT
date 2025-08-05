import gym
from gym import spaces
import numpy as np


class DiscreteBinPackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, container_size=(10, 10, 10), item_sizes=None, grid_resolution=1):
        super().__init__()

        self.container_size = np.array(container_size)
        self.grid_resolution = grid_resolution
        self.grid_size = (self.container_size // self.grid_resolution).astype(int)

        self.max_items = 100
        self.item_sizes = item_sizes or [np.random.randint(1, 4, size=3) for _ in range(self.max_items)]

        # Action space: select one of the discrete grid positions
        self.total_positions = np.prod(self.grid_size)
        self.action_space = spaces.Discrete(self.total_positions)

        # Observation space: [flattened placed items + current item size]
        self.item_dim = 6  # x, y, z, dx, dy, dz
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_items * self.item_dim + 3,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.placed_items = []
        self.current_index = 0
        return self._get_obs(), {}

    def _get_obs(self):
        placed = np.zeros((self.max_items, self.item_dim))
        for i, item in enumerate(self.placed_items):
            placed[i, :] = item
        flat_placed = placed.flatten()
        current_item = self.item_sizes[self.current_index]
        return np.concatenate([flat_placed, current_item / self.container_size], axis=0).astype(np.float32)

    def step(self, action):
        # Decode action to (x, y, z)
        x_idx = action % self.grid_size[0]
        y_idx = (action // self.grid_size[0]) % self.grid_size[1]
        z_idx = action // (self.grid_size[0] * self.grid_size[1])

        pos = np.array([x_idx, y_idx, z_idx]) * self.grid_resolution
        size = self.item_sizes[self.current_index]

        new_box = np.concatenate([pos, size], axis=0)

        if self._check_valid_placement(pos, size):
            self.placed_items.append(new_box)
            reward = np.prod(size)
            done = False
        else:
            reward = -1.0  # Invalid placement
            done = True

        self.current_index += 1
        if self.current_index >= len(self.item_sizes):
            done = True

        return self._get_obs(), reward, done, False, {}

    def _check_valid_placement(self, pos, size):
        max_corner = pos + size
        if np.any(max_corner > self.container_size):
            return False
        for item in self.placed_items:
            other_pos = item[:3]
            other_size = item[3:]
            if self._boxes_overlap(pos, size, other_pos, other_size):
                return False
        return True

    def _boxes_overlap(self, p1, s1, p2, s2):
        return all(
            not (p1[i] + s1[i] <= p2[i] or p2[i] + s2[i] <= p1[i])
            for i in range(3)
        )

    def render(self):
        print(f"Placed {len(self.placed_items)} items.")

    def close(self):
        pass
