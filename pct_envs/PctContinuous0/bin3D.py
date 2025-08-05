import gym
from gym import spaces
import numpy as np

class ContinuousBinPackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, container_size=(10, 10, 10), item_sizes=None, max_items=100):
        super().__init__()

        self.container_size = np.array(container_size)
        self.max_items = max_items

        # Define the action space (x, y, z continuous positions)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Observation space: [placed_items_flat, current_item_size]
        # Each item: [x, y, z, dx, dy, dz]
        self.item_dim = 6
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_items * self.item_dim + 3,),  # current item appended
            dtype=np.float32
        )

        self.item_sizes = item_sizes or [np.random.rand(3) for _ in range(max_items)]
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
        return np.concatenate([flat_placed, current_item], axis=0).astype(np.float32)

    def step(self, action):
        pos = np.clip(action, 0, 1) * self.container_size
        size = self.item_sizes[self.current_index]

        new_box = np.concatenate([pos, size], axis=0)

        if self._check_valid_placement(pos, size):
            self.placed_items.append(new_box)
            reward = np.prod(size)
            done = False
        else:
            reward = -1.0  # Penalty for invalid placement
            done = True    # End episode if invalid

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
